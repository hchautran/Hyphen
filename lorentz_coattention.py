import torch
import torch.nn as nn
import torch.nn.functional as F
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.geoopt.manifolds.stereographic import PoincareBallExact
from hyptorch.geoopt.manifolds.lorentz import math as lmath
from hyptorch.lorentz.layers import LorentzLinear, LorentzAct
from hyptorch.lorentz.layers.LAttn import CrossAttention 
from hyptorch.lorentz.layers.LCLIP import  HypCLIPMLP
from hyptorch.lorentz.blocks.layer_blocks import LorentzBatchNorm1d
from hyptorch.geoopt import ManifoldParameter
import torch.nn.functional as F
from transformers import CLIPTextConfig 
from typing import Optional, Tuple



class CoAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""

    def __init__(
        self,
        manifold=None,
        embedding_dim=100,
        combined_curvature=1,
        fourier=False,
    ):
        super(CoAttention, self).__init__()

        self.manifold = manifold
        self.lorentz = CustomLorentz(k=combined_curvature)
        self.poincare = PoincareBallExact(c=combined_curvature)
        self.embedding_dim = embedding_dim
        self.k = 128
        self.Wl = LorentzLinear(self.lorentz, self.embedding_dim+1, self.embedding_dim+1)
        self.Wc = LorentzLinear(self.lorentz, self.embedding_dim + 1, self.k + 1)
        self.Ws = LorentzLinear(self.lorentz, self.embedding_dim + 1, self.k + 1)
        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.embedding_dim)))
        self.act = LorentzAct(manifold=self.lorentz,activation=nn.Tanh()) 

        # register weights and bias as params
        # self.register_parameter("Wl", self.Wl)
        # self.register_parameter("Wc", self.Wc)
        # self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        # concatenation operation for hyperbolic
        self.register_parameter("concat_m1", self.concat_m1)
        self.register_parameter("concat_m2", self.concat_m2)
        self.register_parameter("concat_b", self.concat_b)

        # initialize data of parameters
        self.Wl.data = torch.randn((self.embedding_dim, self.embedding_dim))
        self.Wc.data = torch.randn((self.k, self.embedding_dim))
        self.Ws.data = torch.randn((self.k, self.embedding_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.concat_m1.data = torch.randn((1, 1))
        self.concat_m2.data = torch.randn((1, 1))
        self.concat_b.data = torch.randn((1, self.embedding_dim))
        self.c = combined_curvature
        self.clip_r = 2.0 
        self.fourier = fourier

    def euclid_to_lorentz(self, x):
        if self.clip_r is not None :
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), self.clip_r / x_norm)
            x = x * fac
        
        x = F.pad(x, (1,0), "constant", 0)
        out = self.lorentz.expmap0(x)
        return out 

 

    def forward(self, sentence_rep, comment_rep):
        """This function will return the shape [batch_size, embedding_dim]."""

        curv = self.c
        if self.fourier:
            # KFU
            sentence_rep = self.poincare.logmap0(sentence_rep)
            assert not torch.isnan(sentence_rep).any(), "sentence is nan before fft2"
            sentence_rep = torch.fft.fft2(sentence_rep).float()
            assert not torch.isnan(sentence_rep).any(), "sentence is nan after fft2"

            comment_rep = self.poincare.logmap0(comment_rep)
            assert not torch.isnan(comment_rep).any(), "comment is nan before fft2"
            comment_rep = torch.fft.fft2(comment_rep).float()
            assert not torch.isnan(comment_rep).any(), "comment is nan after fft2"

            lorentz_sentence_rep = self.euclid_to_lorentz(sentence_rep)
            lorentz_comment_rep = self.euclid_to_lorentz(comment_rep) 
        else:
            lorentz_sentence_rep = lmath.poincare_to_lorentz(sentence_rep, k=curv)
            lorentz_comment_rep = lmath.poincare_to_lorentz(comment_rep, k=curv)


        L = self.Wl(lorentz_comment_rep)
        L = self.act(lorentz_sentence_rep @ L.transpose(-1, -2))
        assert not torch.isnan(L).any(), "L is nan"
        # print(lorentz_comment_rep)

        # Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        Hs_a = self.Ws(lorentz_sentence_rep) 
        Hs_b = self.Wc(lorentz_comment_rep)

        Hs_b = lmath.lorentz_to_poincare(Hs_b, k=curv)
        Hs_a = lmath.lorentz_to_poincare(Hs_a, k=curv)
        # print(Hs_b.shape)

        Hs_b = self.poincare.mobius_matvec(Hs_b.transpose(-1,-2), L)

        Hs = self.poincare.mobius_add(Hs_a, Hs_b)
        Hs = self.poincare.expmap0(torch.tanh(self.poincare.logmap0(Hs)))  # [32, 80, 50]

        # Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L_trans))
        Hc_a = self.Wc(lorentz_comment_rep)
        Hc_b = self.Ws(lorentz_sentence_rep)

        Hc_b = lmath.lorentz_to_poincare(Hc_b, k=curv)
        Hc_a = lmath.lorentz_to_poincare(Hc_a, k=curv)

        Hc_b = self.poincare.mobius_matvec(Hc_b.transpose(-1,-2), L.transpose(-1, -2))
        Hc = self.poincare.mobius_add(Hc_a.transpose(-1,-2), Hc_b.transpose(-1,-2))
        Hc = self.poincare.expmap0(torch.tanh(self.poincare.logmap0(Hc)))  # [32, 80, 10]

        As = self.poincare.mobius_matvec(self.whs, Hs) 
        As = F.softmax(As.transpose(-1, -2), dim=-1)
        assert not torch.isnan(As).any(), "As is nan"

        Ac = self.poincare.mobius_matvec(self.whc, Hc.transpose(-1, -2))
        Ac = F.softmax(Ac.transpose(-1, -2), dim=-1)
        assert not torch.isnan(Ac).any(), "Ac is nan"

        # co_s = torch.matmul(As,sentence_rep) # (1, 100)
        co_s = self.lorentz.centroid(lorentz_sentence_rep, As)
        assert not torch.isnan(co_s).any(), "co_s is nan"

        # co_c = torch.matmul(Ac, comment_rep) # (1, 100)
        co_c = self.lorentz.centroid(lorentz_comment_rep, Ac)
        assert not torch.isnan(co_c).any(), "co_c is nan"

        co_sc = self.lorentz.concat(co_s, co_c)
        co_sc = torch.squeeze(co_sc)

        assert not torch.isnan(co_sc).any(), "co_sc is nan"
        return co_sc, As, Ac  # [32, 200],


class CrossAttentionEncoder(nn.Module):
    def __init__(self, manifold:CustomLorentz, config: CLIPTextConfig):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.cross_attentkon= CrossAttention(manifold, config)
        self.self_attention= CrossAttention(manifold, config)
        self.batch_norm1 = LorentzBatchNorm1d(manifold, self.embed_dim+1)
        self.batch_norm2 = LorentzBatchNorm1d(manifold, self.embed_dim+1)
        self.mlp = HypCLIPMLP(manifold, config)
        self.manifold = manifold

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = x 

        hidden_states = self.manifold.projx(self.batch_norm1(x))
        hidden_states, attn_weights = self.cross_attentkon(
            x, y,
            output_attentions=output_attentions,
        )
        hidden_states, attn_weights = self.self_attention(
            hidden_states, hidden_states,
            output_attentions=output_attentions,
        )
        self.manifold.assert_check_point_on_manifold(hidden_states)
        
        # residual connection
        hidden_states =  self.manifold.projx(self.manifold.lorentz_addition(hidden_states, residual))


        residual = hidden_states

        hidden_states = self.manifold.projx(self.batch_norm2(hidden_states))
        # self.manifold.assert_check_point_on_manifold(hidden_states)
        hidden_states = self.mlp(hidden_states)
        # self.manifold.assert_check_point_on_manifold(hidden_states)

        # residual connection
        hidden_states = self.manifold.projx(self.manifold.lorentz_addition(hidden_states, residual))
        # self.manifold.assert_check_point_on_manifold(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

 
class CoCrossAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""
    
    def __init__(self, manifold:CustomLorentz , embedding_dim = 100, fourier = False):
        super(CoCrossAttention, self).__init__()

        self.k = 128
        self.manifold = manifold
        self.embedding_dim = embedding_dim
        config = CLIPTextConfig(
            hidden_size=self.embedding_dim, 
            num_attention_heads=1
        )
        self.cross_attn_s = CrossAttentionEncoder(manifold, config)
        self.cross_attn_c = CrossAttentionEncoder(manifold, config)
        self.w_hs = ManifoldParameter(manifold=manifold, data=manifold.random_normal(1, self.embedding_dim+1))
        self.w_hc = ManifoldParameter(manifold=manifold, data=manifold.random_normal(1, self.embedding_dim+1))
        self.register_parameter("w_hs", self.w_hs)
        self.register_parameter("w_hc", self.w_hc)


    def forward(self, sentence_rep, comment_rep):
        """This function will return the shape [batch_size, embedding_dim]."""

        s = self.cross_attn_s(sentence_rep, comment_rep)[0]
        c = self.cross_attn_c(comment_rep, sentence_rep)[0]
        s = torch.squeeze(self.manifold.centroid(s))
        c = torch.squeeze(self.manifold.centroid(c))
        
        co_sc = self.manifold.projx(self.manifold.concat(s,c))
        self.manifold.assert_check_point_on_manifold(co_sc)
        return co_sc

   