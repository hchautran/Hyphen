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
from ..utils.utils import matrix_mul
from typing import Union
from .s4 import FFTConv, DropoutNd 


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
        bidirectional=False,
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
                FFTConv(
                    d_model=d_model, 
                    d_state=d_state, 
                    dropout=dropout, 
                    transposed=True,
                    lr=min(0.0001, 0.01),
                    mode='diag', 
                    init='diag-lin', 
                    bidirectional=bidirectional, 
                    disc='zoh', 
                    real_transform='exp'
                )
            )
            self.norms.append(nn.LayerNorm(d_model))
            self.dropouts.append(DropoutNd(dropout))
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model*factor),
            nn.GELU(),
            nn.Linear(d_model*factor, d_output),
        ) 


    def forward(self, x:torch.Tensor, pooling=False):
        """
        Input x is shape (B, L, d_input)
        """
        x = self.encoder(x)  # (B, L, d_input) -> (B, L, d_model)

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
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
        self.word_ssm= S4Model(d_input=word_hidden_size, d_model=word_hidden_size//2, d_output=sent_hidden_size, n_layers=1, prenorm=False, d_state=word_hidden_size//2, factor=factor, bidirectional=True)
        self.sent_ssm= S4Model(d_input=sent_hidden_size, d_model=sent_hidden_size//2, d_output=sent_hidden_size, n_layers=1, prenorm=False, d_state=sent_hidden_size//2, factor=factor, bidirectional=False)
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