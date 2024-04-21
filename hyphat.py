import torch
import torch.nn as nn
import torch.nn as nn

from lcoattention import CoAttention 
from hypComEnc import HypComEnc
from hyphatpost import HyphatPost
from utils.layers.hyp_layers import *
from utils import manifolds
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LMLR import LorentzMLR 
from hyptorch.geoopt.manifolds.lorentz import math as lmath 
from hyp_math import E2Lorentz, P2Lorentz

class Hyphat(nn.Module):

    def __init__(
        self, 
        embedding_matrix,  
        word_hidden_size, 
        sent_hidden_size, 
        max_sent_length, 
        max_word_length, 
        device, 
        graph_hidden, 
        num_classes = 2, 
        max_sentence_count = 50 , 
        max_comment_count = 10, 
        batch_size = 32 ,
        embedding_dim = 100, 
        latent_dim = 100, 
        graph_glove_dim = 100, 
        manifold = "hyper", 
        use_gat=False,
        content_module =True, 
        comment_module = True, 
        fourier = False
    ):
        super(Hyphat,self).__init__()

        self.device = device
        self.fourier = fourier
        self.comment_curvature = torch.tensor(1.0)
        self.content_curvature = torch.tensor(1.0)
        self.combined_curvature = torch.tensor(1.0)
        self.graph_glove_dim = graph_glove_dim#the dimension of glove embeddings used to initialise the comments amr graph
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_sentence_count = max_sentence_count
        self.max_comment_count = max_comment_count
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length  = max_word_length 
        self.graph_hidden = graph_hidden
        # self.poincare = getattr(manifolds, manifold)()
        self.manifold = getattr(manifolds, 'PoincareBall')()
        self.lorentz = CustomLorentz(k=1)
        self.comment_module = comment_module
        self.content_module = content_module 

        self.content_encoder= HyphatPost(
            word_hidden_size=self.word_hidden_size, 
            sent_hidden_size=self.sent_hidden_size, 
            batch_size=batch_size, 
            num_classes=num_classes, 
            embedding_matrix=embedding_matrix, 
            max_sent_length=self.max_sent_length, 
            max_word_length=self.max_word_length, 
            device=self.device, 
            manifold=self.manifold, 
            content_curvature=self.content_curvature
        )
        self.comment_encoder = HypComEnc(
            self.graph_glove_dim, 
            self.graph_hidden, 
            num_classes, 
            self.max_comment_count, 
            device=self.device, 
            manifold = self.manifold, 
            content_module = self.content_module, 
            comment_curvature = self.comment_curvature, 
            use_gat=use_gat
        )
        self.coattention = CoAttention(manifold=self.manifold, embedding_dim=self.embedding_dim, fourier=self.fourier, combined_curvature=self.combined_curvature)
        
        if self.comment_module and self.content_module: 
            self.fc = nn.Sequential( 
                # LFC_Block(self.manifold, 2*latent_dim + 1, 2*latent_dim + 1, normalization="batch_norm"),
                LorentzMLR(self.lorentz, 2*latent_dim + 1, num_classes),
            )
        elif self.comment_module: 
            self.fc = nn.Sequential( 
                # LFC_Block(self.manifold, latent_dim + 1, latent_dim + 1, normalization="batch_norm"),
                LorentzMLR(self.lorentz, latent_dim + 1, num_classes),
            )
        else: 
            self.fc = nn.Sequential( 
                LorentzMLR(self.lorentz, latent_dim + 1, num_classes),
            )

    def forward(self, content, comment, subgraphs):
        
        #both content and comments modules are on
        content_embedding = self.content_encoder(content)
        comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
        assert not torch.isnan(content_embedding).any(), "content_embedding is nan"
        assert not torch.isnan(comment_embedding).any(), "comment_embedding is nan"


        coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
        preds = self.fc(coatten)
        return preds

