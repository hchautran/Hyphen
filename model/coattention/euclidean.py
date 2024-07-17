import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.layers.hyp_layers import HypAct


class CoAttention(nn.Module):

    """This is the class for Hyperbolic Fourier-coattention mechanism."""

    def __init__(
        self,
        manifold=None,
        latent_dim=100,
        embedding_dim=100,
        fourier=False,
    ):
        super(CoAttention, self).__init__()
        self.manifold = manifold

        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim
        self.k = 128
        self.Wl = nn.Parameter(torch.Tensor((self.latent_dim, self.latent_dim)))
        self.Wc = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.Ws = nn.Parameter(torch.Tensor((self.k, self.latent_dim)))
        self.whs = nn.Parameter(torch.Tensor((1, self.k)))
        self.whc = nn.Parameter(torch.Tensor((1, self.k)))
        self.concat_m1 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_m2 = nn.Parameter(torch.Tensor((1, 1)))
        self.concat_b = nn.Parameter(torch.Tensor((1, self.embedding_dim)))

        # register weights and bias as params
        self.register_parameter("Wl", self.Wl)
        self.register_parameter("Wc", self.Wc)
        self.register_parameter("Ws", self.Ws)
        self.register_parameter("whs", self.whs)
        self.register_parameter("whc", self.whc)

        # concatenation operation for hyperbolic
        self.register_parameter("concat_m1", self.concat_m1)
        self.register_parameter("concat_m2", self.concat_m2)
        self.register_parameter("concat_b", self.concat_b)

        # initialize data of parameters
        self.Wl.data = torch.randn((self.latent_dim, self.latent_dim))
        self.Wc.data = torch.randn((self.k, self.latent_dim))
        self.Ws.data = torch.randn((self.k, self.latent_dim))
        self.whs.data = torch.randn((1, self.k))
        self.whc.data = torch.randn((1, self.k))
        self.concat_m1.data = torch.randn((1, 1))
        self.concat_m2.data = torch.randn((1, 1))
        self.concat_b.data = torch.randn((1, self.embedding_dim))
        self.fourier = fourier

    def forward(self, sentence_rep, comment_rep):
        """This function will return the shape [batch_size, embedding_dim]."""
        if self.fourier:
            # KFU
            sentence_rep = torch.fft.fft2(sentence_rep).float()
            comment_rep = torch.fft.fft2(comment_rep).float()

        sentence_rep_trans = sentence_rep.transpose(2, 1)  # [32, 100, 50]
        comment_rep_trans = comment_rep.transpose(2, 1)  # [32, 100, 10]


        L = torch.tanh(torch.matmul(torch.matmul(comment_rep, self.Wl), sentence_rep_trans))
        Hs = torch.tanh(torch.matmul(self.Ws, sentence_rep_trans) + torch.matmul(torch.matmul(self.Wc, comment_rep_trans), L))
        Hc = torch.tanh(torch.matmul(self.Wc, comment_rep_trans)+ torch.matmul(torch.matmul(self.Ws, sentence_rep_trans), L.transpose(-1,-2)))
        As = F.softmax(torch.matmul(self.whs, Hs), dim=2)
        Ac = F.softmax(torch.matmul(self.whc, Hc), dim=2)
        co_s = torch.matmul(As,sentence_rep) # (1, 100)
        co_c = torch.matmul(Ac, comment_rep) # (1, 100)
        co_sc = torch.cat([co_s, co_c], dim=-1)
        co_sc = torch.squeeze(co_sc)

        # assert not torch.isnan(co_sc).any(), "co_sc is nan"
        return co_sc, As, Ac  # [32, 200],
