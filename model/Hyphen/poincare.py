import torch
import torch.nn as nn
from ..coattention.poincare import CoAttention
from ..comenc.poincare_gcn import ComEnc 
from ..postenc.poincare_han import PostEnc 
from ..utils.layers.hyp_layers import *
from geoopt import PoincareBall 
from hyptorch.geoopt import PoincareBall 
from hyptorch.poincare.layers import PMLR 


class Hyphen(nn.Module):

    def __init__(
        self, 
        manifold:PoincareBall,
        embedding_matrix,  
        word_hidden_size, 
        sent_hidden_size, 
        device, 
        graph_hidden, 
        num_classes = 2, 
        max_comment_count = 10, 
        batch_size = 32 ,
        embedding_dim = 100, 
        latent_dim = 100, 
        graph_glove_dim = 100, 
        fourier = False, 
    ):

        super(Hyphen,self).__init__()
        self.comment_curvature = torch.tensor(1.0)
        self.content_curvature = torch.tensor(1.0)
        self.combined_curvature = torch.tensor(1.0)
        self.fourier = fourier
        self.graph_glove_dim = graph_glove_dim#the dimension of glove embeddings used to initialise the comments amr graph
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_comment_count = max_comment_count
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.graph_hidden = graph_hidden
        self.manifold = manifold 
        print('building HypPostEnc')
        self.content_encoder= PostEnc(
            word_hidden_size=self.word_hidden_size, 
            sent_hidden_size=self.sent_hidden_size, 
            batch_size=batch_size, 
            num_classes=num_classes, 
            embedding_matrix=embedding_matrix, 
            device=self.device, 
            manifold=self.manifold, 
        )
        print('building HypComEnc')
        self.comment_encoder = ComEnc(
            in_dim=self.graph_glove_dim, 
            hidden_dim=self.graph_hidden, 
            n_classes=num_classes, 
            max_comment_count=self.max_comment_count, 
            device=self.device, 
            manifold=self.manifold, 
            content_module=self.content_module, 
        )
        print('building CoAttention')
        self.coattention = CoAttention(
            embedding_dim=embedding_dim, 
            latent_dim=latent_dim, 
            manifold=self.manifold,  
            fourier=self.fourier
        )
        
        self.fc = nn.Sequential(
            # HypLinear(manifold=self.manifold, in_features=2*latent_dim, out_features=2*latent_dim,c=self.manifold.c, use_bias=True, dropout=0.1),
            # PMLR.UnidirectionalPoincareMLR(2*latent_dim, num_classes, ball=self.manifold)
            nn.Linear(2*latent_dim, num_classes)
        )

    def hir_loss(self, embeddings):
        # regularization on the tangent distance to the origin without changing original embeddings
        embeddings_tan = self.manifold.logmap0(embeddings)
        embeddings_tan = embeddings_tan - embeddings_tan.mean(dim=0)  # Equation (7)
        tangent_mean_norm = (1e-6 + embeddings_tan.pow(2).sum(dim=1).mean())
        tangent_mean_norm = -tangent_mean_norm
        return (max(tangent_mean_norm, -10) + 10)

    def forward(self, content, comment, subgraphs):
        
        #both content and comments modules are on

            #hyphen-euclidean 
        _, content_embedding = self.content_encoder(content)
        comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
        
        assert not torch.isnan(content_embedding).any(), "content_embedding is nan"
        assert not torch.isnan(comment_embedding).any(), "comment_embedding is nan"
        coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
        preds = self.fc(coatten)
        if torch.isnan(preds).any():
            print(preds, coatten)
            preds = torch.nan_to_num(preds, nan = 0.0)

        assert not torch.isnan(preds).any(), "preds is nan"
        return preds, As, Ac
        #only comment module is on
     