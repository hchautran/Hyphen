import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt.manifolds.stereographic import PoincareBall 
from hyptorch.geoopt.manifolds.lorentz import math as lmath
from hyptorch.lorentz.layers import LorentzLinear, LorentzAct
from hyptorch.geoopt import ManifoldParameter 
import torch.nn.functional as F



class CoAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""

    def __init__(
        self,
        manifold:CustomLorentz,
        embedding_dim=100,
        latent_dim=100,
        fourier=True,
    ):
        super(CoAttention, self).__init__()

        self.manifold = manifold
        print('lorentz', self.manifold.k.item())
        self.poincare = PoincareBall(c=self.manifold.k.item())
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.k = 128
        self.Wl = LorentzLinear(self.manifold, self.latent_dim+1, self.embedding_dim+1)
        self.Wc = LorentzLinear(self.manifold, self.latent_dim + 1, self.k + 1)
        self.Ws = LorentzLinear(self.manifold, self.latent_dim+ 1, self.k + 1)
        self.whs = ManifoldParameter(torch.Tensor((1, self.k)) )
        self.whc = ManifoldParameter(torch.Tensor((1, self.k)) )
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.latent_dim)))
        self.gelu = nn.GELU()
        self.act = LorentzAct(manifold=self.manifold, activation=nn.GELU()) 
        

        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        # concatenation operation for hyperbolic

        # initialize data of parameters
        self.whs.data = torch.rand((1, self.k))
        self.whc.data = torch.rand((1, self.k))
        self.clip_r = 2.0 
        self.fourier = fourier

    def euclid_to_lorentz(self, x):
        if self.clip_r is not None :
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm)
            x = x * fac
        
        x = F.pad(x, (1,0), "constant", 0)
        out = self.manifold.expmap0(x)
        return out 

 

    def forward(self, sentence_rep:torch.Tensor, comment_rep:torch.Tensor):
        """This function will return the shape [batch_size, embedding_dim]."""
        # self.poincare.assert_check_point_on_manifold(comment_rep)
        # self.poincare.assert_check_point_on_manifold(sentence_rep)

        curv = 1.0 
        if self.fourier:
            # KFU
            assert not torch.isnan(sentence_rep).any(), "sentence is nan before fft2"
            sentence_rep = torch.fft.fft2(sentence_rep).float()
            assert not torch.isnan(sentence_rep).any(), "sentence is nan after fft2"

            comment_rep = self.poincare.logmap0(comment_rep)
            assert not torch.isnan(comment_rep).any(), "comment is nan before fft2"
            comment_rep = torch.fft.fft2(comment_rep).float()
            assert not torch.isnan(comment_rep).any(), "comment is nan after fft2"

            lorentz_sentence_rep = self.euclid_to_lorentz(sentence_rep)
            lorentz_comment_rep = self.euclid_to_lorentz(comment_rep) 
            # self.manifold.assert_check_point_on_manifold(lorentz_comment_rep)
            # self.manifold.assert_check_point_on_manifold(lorentz_sentence_rep)
        else:
            lorentz_sentence_rep = lmath.poincare_to_lorentz(sentence_rep, k=curv)
            lorentz_comment_rep = lmath.poincare_to_lorentz(comment_rep, k=curv)


        L = self.Wl(lorentz_comment_rep)
        L = self.act(lorentz_sentence_rep @ L.transpose(-1, -2))
        assert not torch.isnan(L).any(), "L is nan"

        # Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        Hs_a = self.Ws(lorentz_sentence_rep) 
        Hs_b = self.Wc(lorentz_comment_rep)

        Hs_b = lmath.lorentz_to_poincare(Hs_b, k=self.manifold.k)
        Hs_a = lmath.lorentz_to_poincare(Hs_a, k=self.manifold.k)
        # print(Hs_b.shape)

        Hs_b = self.poincare.mobius_matvec(Hs_b.transpose(-1,-2), L)

        Hs = self.poincare.mobius_add(Hs_a, Hs_b)
        Hs = self.poincare.expmap0(self.gelu(self.poincare.logmap0(Hs)))  # [32, 80, 50]

        # Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        Hc_a = self.Wc(lorentz_comment_rep)
        Hc_b = self.Ws(lorentz_sentence_rep)

        Hc_b = lmath.lorentz_to_poincare(Hc_b, k=self.manifold.k)
        Hc_a = lmath.lorentz_to_poincare(Hc_a, k=self.manifold.k)

        Hc_b = self.poincare.mobius_matvec(Hc_b.transpose(-1,-2), L.transpose(-1, -2))
        Hc = self.poincare.mobius_add(Hc_a.transpose(-1,-2), Hc_b.transpose(-1,-2))
        Hc = self.poincare.expmap0(self.gelu(self.poincare.logmap0(Hc)))  # [32, 80, 10]

        As = self.poincare.mobius_matvec(self.whs, Hs) 
        As = F.softmax(As.transpose(-1, -2), dim=-1)
        assert not torch.isnan(As).any(), "As is nan"

        Ac = self.poincare.mobius_matvec(self.whc, Hc.transpose(-1, -2))
        Ac = F.softmax(Ac.transpose(-1, -2), dim=-1)
        assert not torch.isnan(Ac).any(), "Ac is nan"

        # co_s = torch.matmul(As,sentence_rep) # (1, 100)
        co_s = self.manifold.centroid(lorentz_sentence_rep, As)
        assert not torch.isnan(co_s).any(), "co_s is nan"

        # co_c = torch.matmul(Ac, comment_rep) # (1, 100)
        co_c = self.manifold.centroid(lorentz_comment_rep, Ac)
        assert not torch.isnan(co_c).any(), "co_c is nan"

        co_sc = self.manifold.concat(co_s, co_c)
        co_sc = torch.squeeze(co_sc)

        assert not torch.isnan(co_sc).any(), "co_sc is nan"
        return co_sc, As, Ac  # [32, 200],

