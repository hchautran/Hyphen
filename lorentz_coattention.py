import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt.manifolds.lorentz import math as lmath
from hyptorch.lorentz.layers import LorentzLinear, LorentzAct
from hyptorch.geoopt import ManifoldParameter
import torch.nn.functional as F

class CoAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, manifold:CustomLorentz , embedding_dim = 100, fourier = False):
        super(CoAttention, self).__init__()

        self.k = 128
        self.manifold = manifold
        self.embedding_dim = embedding_dim
        self.w_l = LorentzLinear(manifold, self.embedding_dim + 1, self.embedding_dim + 1, normalize=True)
        self.w_c = LorentzLinear(manifold, self.embedding_dim + 1, self.k + 1, normalize=True)
        self.w_s = LorentzLinear(manifold, self.embedding_dim + 1, self.k + 1, normalize=True)
        self.w_hs = ManifoldParameter(manifold=manifold, data=manifold.random_normal(1, self.k+1))
        self.w_hc = ManifoldParameter(manifold=manifold, data=manifold.random_normal(1, self.k+1))

        self.c = self.manifold.k 
        self.act = LorentzAct(manifold=manifold, activation=nn.Tanh())

        #concatenation operation for hyperbolic 
        self.register_parameter("w_hs", self.w_hs)
        self.register_parameter("w_hc", self.w_hc)

        #initialize data of parameters
        self.fourier = fourier


    def forward(self, sentence_rep, comment_rep):
        """This function will return the shape [batch_size, embedding_dim]."""

        if self.fourier:
            # KFU
            sentence_rep = self.manifold.logmap0(sentence_rep) 
            sentence_rep = torch.fft.fft2(sentence_rep).float()

            comment_rep = self.manifold.logmap0(comment_rep)
            comment_rep = torch.fft.fft2(comment_rep).float()

            sentence_rep = self.manifold.expmap0(sentence_rep)
            comment_rep = self.manifold.expmap0(comment_rep)


        C =2 + 2*self.manifold.matmul(comment_rep, self.w_l(sentence_rep).permute(0,2,1))

        H_s = self.act(self.manifold.lorentz_addition(
            self.w_s(sentence_rep), 
            self.manifold.centroid(self.w_c(comment_rep), F.softmax(C.permute(0,2,1), dim=-1)))
        ) # B x k x 196
        self.manifold.assert_check_point_on_manifold(H_s)

        H_c = self.act(self.manifold.lorentz_addition(
            self.w_c(comment_rep), 
            self.manifold.centroid(self.w_s(sentence_rep), F.softmax(C, dim=-1))
        ))
        self.manifold.assert_check_point_on_manifold(H_c)

        a_c = F.softmax(2 + 2*self.manifold.matmul(self.w_hc, H_c.transpose(-2,-1)), dim=-1)
        a_s = F.softmax(2 + 2*self.manifold.matmul(self.w_hs, H_s.transpose(-2,-1)), dim=-1)
        # print(a_s)
        # print(a_c)
 
        s = torch.squeeze(self.manifold.centroid(sentence_rep, a_s))
        c = torch.squeeze(self.manifold.centroid(comment_rep, a_c))

        self.manifold.assert_check_point_on_manifold(s)
        self.manifold.assert_check_point_on_manifold(c)
        co_sc = self.manifold.concat(s,c)
        self.manifold.assert_check_point_on_manifold(co_sc)
        return co_sc, a_s, a_c
  