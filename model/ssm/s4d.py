"""Minimal version of S4D with extra options and features stripped out, for pedagogical purposes."""

import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch
import torch.nn as nn
from ..coattention.poincare import CoAttention as PoincareCoAttn
from ..coattention.euclidean import CoAttention as EuclidCoAttn
from ..coattention.lorentz import CoAttention as LorentzCoAttn
from ..utils.layers.hyp_layers import *
from hyptorch.geoopt import PoincareBall, Euclidean
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers import LorentzMLR 
from ..utils.utils import matrix_mul, element_wise_mul
from typing import Union


if tuple(map(int, torch.__version__.split('.')[:2])) == (1, 11):
    print("WARNING: Dropout is bugged in PyTorch 1.11. Results may be worse.")
    dropout_fn = nn.Dropout
if tuple(map(int, torch.__version__.split('.')[:2])) >= (1, 12):
    dropout_fn = nn.Dropout1d
else:
    dropout_fn = nn.Dropout2d



class DropoutNd(nn.Module):
    def __init__(self, p: float = 0.5, tie=True, transposed=True):
        """
        tie: tie dropout mask across sequence lengths (Dropout1d/2d/3d)
        """
        super().__init__()
        if p < 0 or p >= 1:
            raise ValueError("dropout probability has to be in [0, 1), " "but got {}".format(p))
        self.p = p
        self.tie = tie
        self.transposed = transposed
        self.binomial = torch.distributions.binomial.Binomial(probs=1-self.p)

    def forward(self, X):
        """X: (batch, dim, lengths...)."""
        if self.training:
            if not self.transposed: X = rearrange(X, 'b ... d -> b d ...')
            # binomial = torch.distributions.binomial.Binomial(probs=1-self.p) # This is incredibly slow because of CPU -> GPU copying
            mask_shape = X.shape[:2] + (1,)*(X.ndim-2) if self.tie else X.shape
            # mask = self.binomial.sample(mask_shape)
            mask = torch.rand(*mask_shape, device=X.device) < 1.-self.p
            X = X * mask * (1.0/(1-self.p))
            if not self.transposed: X = rearrange(X, 'b d ... -> b ... d')
            return X
        return X

class S4DKernel(nn.Module):
    """Generate convolution kernel from diagonal SSM parameters."""

    def __init__(self, d_model, N=64, dt_min=0.001, dt_max=0.1, lr=None):
        super().__init__()
        # Generate dt
        H = d_model
        log_dt = torch.rand(H) * (
            math.log(dt_max) - math.log(dt_min)
        ) + math.log(dt_min)

        C = torch.randn(H, N // 2, dtype=torch.cfloat)
        self.C = nn.Parameter(torch.view_as_real(C))
        self.register("log_dt", log_dt, lr)

        log_A_real = torch.log(0.5 * torch.ones(H, N//2))
        A_imag = math.pi * repeat(torch.arange(N//2), 'n -> h n', h=H)
        self.register("log_A_real", log_A_real, lr)
        self.register("A_imag", A_imag, lr)

    def forward(self, L):
        """
        returns: (..., c, L) where c is number of channels (default 1)
        """

        # Materialize parameters
        dt = torch.exp(self.log_dt) # (H)
        C = torch.view_as_complex(self.C) # (H N)
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag # (H N)

        # Vandermonde multiplication
        dtA = A * dt.unsqueeze(-1)  # (H N)
        K = dtA.unsqueeze(-1) * torch.arange(L, device=A.device) # (H N L)
        C = C * (torch.exp(dtA)-1.) / A
        K = 2 * torch.einsum('hn, hnl -> hl', C, torch.exp(K)).real

        return K

    def register(self, name, tensor, lr=None):
        """Register a tensor with a configurable learning rate and 0 weight decay"""

        if lr == 0.0:
            self.register_buffer(name, tensor)
        else:
            self.register_parameter(name, nn.Parameter(tensor))

            optim = {"weight_decay": 0.0}
            if lr is not None: optim["lr"] = lr
            setattr(getattr(self, name), "_optim", optim)


class S4D(nn.Module):
    def __init__(self, d_model, d_state=64, dropout=0.0, transposed=True, **kernel_args):
        super().__init__()

        self.h = d_model
        self.n = d_state
        self.d_output = self.h
        self.transposed = transposed

        self.D = nn.Parameter(torch.randn(self.h))

        # SSM Kernel
        self.kernel = S4DKernel(self.h, N=self.n, **kernel_args)

        # Pointwise
        self.activation = nn.GELU()
        # dropout_fn = nn.Dropout2d # NOTE: bugged in PyTorch 1.11
        dropout_fn = DropoutNd
        self.dropout = dropout_fn(dropout) if dropout > 0.0 else nn.Identity()

        # position-wise output transform to mix features
        self.output_linear = nn.Sequential(
            nn.Conv1d(self.h, 2*self.h, kernel_size=1),
            nn.GLU(dim=-2),
        )

    def forward(self, u, **kwargs): # absorbs return_output and transformer src mask
        """ Input and output shape (B, H, L) """
        if not self.transposed: u = u.transpose(-1, -2)
        L = u.size(-1)

        # Compute SSM Kernel
        k = self.kernel(L=L) # (H L)

        # Convolution
        k_f = torch.fft.rfft(k, n=2*L) # (H L)
        u_f = torch.fft.rfft(u, n=2*L) # (B H L)
        y = torch.fft.irfft(u_f*k_f, n=2*L)[..., :L] # (B H L)

        # Compute D term in state space equation - essentially a skip connection
        y = y + u * self.D.unsqueeze(-1)

        y = self.dropout(self.activation(y))
        y = self.output_linear(y)
        if not self.transposed: y = y.transpose(-1, -2)
        return y, None # Return a dummy state to satisfy this repo's interface, but this can be modified

        
        

class S4Model(nn.Module):

    def __init__(
        self,
        d_input=100,
        d_model=256,
        d_output=100,
        n_layers=1,
        d_state=64,
        dropout=0.1,
        prenorm=False,
        factor=2
    ):
        super().__init__()

        self.prenorm = prenorm

        # Linear encoder (d_input = 1 for grayscale and 3 for RGB)
        self.encoder = nn.Linear(d_input, d_model)

        # Stack S4 layers as residual blocks
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(d_model, d_state=d_state, dropout=dropout, transposed=True, lr=min(0.0001, 0.01))
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(dropout_fn(dropout))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*factor),
            nn.GELU(),
            nn.Linear(d_model*factor, d_output),
        ) 


    def forward(self, x:torch.Tensor, pooling=False):
        """
        Input x is shape (B, L, d_input)
        """
        # print(x.shape)
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)
            z = x
            z, _ = layer(z)
            z = dropout(z)
            x = z + x
            x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)
        x = self.decoder(x)  #
        if pooling:
            return x.mean(dim=1)
        return x


class S4DEnc(nn.Module):
    def __init__(
        self, 
        manifold,
        word_hidden_size, 
        sent_hidden_size, 
        embedding_matrix, 
        factor=2,
        pooling_mode = 'mean'
    ):
        super(S4DEnc, self).__init__()
        self.manifold = manifold
        self.pooling_mode = pooling_mode
        self.word_ssm= S4Model(d_input=word_hidden_size, d_model=word_hidden_size//2, d_output=sent_hidden_size, n_layers=1, prenorm=False, d_state=word_hidden_size//2, factor=factor)
        self.sent_ssm= S4Model(d_input=sent_hidden_size, d_model=sent_hidden_size//2, d_output=sent_hidden_size, n_layers=1, prenorm=False, d_state=sent_hidden_size//2, factor=factor)
        self.word_weight= nn.Parameter(torch.Tensor(word_hidden_size, word_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(word_hidden_size, 1))
        self.lookup = self.create_embeddeding_layer(embedding_matrix)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input):
        output_list = []
        # input = input.permute(1, 0, 2)
        for x in input:
            x = self.lookup(x)
            if self.pooling_mode == 'mean': 
                x = self.word_ssm(x=x, pooling=True) 
            else:
                x = self.word_ssm(x=x, pooling=False) 
                output = matrix_mul(x, self.word_weight, self.word_bias)
                output = matrix_mul(x, self.context_weight).permute(1,0)[..., None]
                output = F.softmax(output, dim=-1)
                x = (x.transpose(-1,-2) @ output).squeeze(-1)
            output_list.append(x)

        output = torch.stack(output_list, dim=0)
        x = self.sent_ssm(output)
        if not isinstance(self.manifold, Euclidean):
            clip_r = 2.0 
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), clip_r/ x_norm)
            x = x * fac
            output = self.manifold.expmap0(x)

        return output 


    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer


class SSM4RC(nn.Module):

    def __init__(
        self, 
        manifold:Union[PoincareBall, CustomLorentz],
        embedding_matrix,  
        word_hidden_size, 
        sent_hidden_size, 
        device, 
        graph_hidden, 
        num_classes = 2, 
        batch_size = 32 ,
        embedding_dim = 100, 
        latent_dim = 100, 
        graph_glove_dim = 100, 
        content_module =True, 
        comment_module = True, 
        fourier = False, 
    ):

        super(SSM4RC,self).__init__()
        self.comment_curvature = torch.tensor(1.0)
        self.content_curvature = torch.tensor(1.0)
        self.combined_curvature = torch.tensor(1.0)
        self.fourier = fourier
        self.graph_glove_dim = graph_glove_dim#the dimension of glove embeddings used to initialise the comments amr graph
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.graph_hidden = graph_hidden
        self.manifold = manifold 
        self.comment_module = comment_module
        self.content_module = content_module 
        print('building HypPostEnc')
        self.content_encoder= S4DEnc(
            manifold=self.manifold, 
            word_hidden_size=word_hidden_size*2,
            sent_hidden_size=sent_hidden_size*2,
            embedding_matrix=embedding_matrix
        )
        print('building HypComEnc')
        self.comment_encoder= S4DEnc(
            manifold=self.manifold, 
            word_hidden_size=word_hidden_size*2,
            sent_hidden_size=sent_hidden_size*2,
            embedding_matrix=embedding_matrix
        )
        print('building CoAttention')

        if isinstance(self.manifold, CustomLorentz):
            self.coattention = LorentzCoAttn(
                embedding_dim=embedding_dim, 
                latent_dim=latent_dim, 
                manifold=self.manifold,  
                fourier=self.fourier
            )
            self.fc = LorentzMLR(self.manifold, num_features=2*latent_dim+1, num_classes=2)
            # self.fc =  nn.Linear(2*latent_dim+1, num_classes)
        elif isinstance(self.manifold, PoincareBall):
            self.coattention = PoincareCoAttn(
                embedding_dim=embedding_dim, 
                latent_dim=latent_dim, 
                manifold=self.manifold,  
                fourier=self.fourier
            )
            self.fc =  nn.Linear(2*latent_dim, num_classes)
        else:
            self.coattention = EuclidCoAttn(
                embedding_dim=embedding_dim, 
                latent_dim=latent_dim, 
                fourier=self.fourier
            )
            self.fc =  nn.Linear(2*latent_dim, num_classes)


    def forward(self, content, comment):
        
        content_embedding = self.content_encoder(content)
        comment_embedding = self.content_encoder(comment)
        coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
        preds = self.fc(coatten)
        return preds, As, Ac
     