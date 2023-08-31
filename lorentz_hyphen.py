import torch
import torch.nn as nn
import torch.nn as nn

from lorentz_coattention import CoAttention 
from hypComEnc import HypComEnc
from hypPostEnc import HypPostEnc
from utils.layers.hyp_layers import *
from utils import manifolds
from utils.manifolds import Euclidean
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LFC import LorentzLinear 
from hyptorch.lorentz.layers.LMLR import LorentzMLR 
from hyptorch.geoopt.manifolds.lorentz import math as lmath 

class Hyphen(nn.Module):

    def __init__(self, embedding_matrix,  word_hidden_size, sent_hidden_size, max_sent_length, max_word_length, device, graph_hidden, num_classes = 2, max_sentence_count = 50 , max_comment_count = 10, batch_size = 32 ,embedding_dim = 100, latent_dim = 100, graph_glove_dim = 100, manifold = "hyper",
    content_module =True, comment_module = True, fourier = False):

        super(Hyphen,self).__init__()
        self.comment_curvature = torch.tensor(1.0)
        self.content_curvature = torch.tensor(1.0)
        self.combined_curvature = torch.tensor(1.0)
        self.fourier = fourier
        self.graph_glove_dim = graph_glove_dim#the dimension of glove embeddings used to initialise the comments amr graph
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_sentence_count = max_sentence_count
        self.max_comment_count = max_comment_count
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length  = max_word_length 
        self.graph_hidden = graph_hidden
        self.poincare = getattr(manifolds, manifold)()
        self.manifold = CustomLorentz(k=self.combined_curvature, atol=1e-2, rtol=1e-2) 
        self.comment_module = comment_module
        self.content_module = content_module 
        print('building HypPostEnc')
        self.content_encoder= HypPostEnc(self.word_hidden_size, self.sent_hidden_size, batch_size, num_classes, embedding_matrix, self.max_sent_length, self.max_word_length, self.device, manifold = self.poincare , content_curvature = self.content_curvature)
        print('building HypComEnc')
        self.comment_encoder = HypComEnc(self.graph_glove_dim, self.graph_hidden, num_classes, self.max_comment_count, device= self.device, manifold = self.poincare, content_module = self.content_module, comment_curvature = self.comment_curvature)
        print('building CoAttention')
        self.coattention = CoAttention(manifold=self.manifold, embedding_dim=self.embedding_dim, fourier=False)
        
        if self.comment_module and self.content_module: 
            self.fc = nn.Sequential( 
                # LorentzLinear(self.manifold, 2*latent_dim + 1, 2*latent_dim + 1),
                LorentzMLR(self.manifold, 2*latent_dim + 1, num_classes),
            )
        elif self.comment_module: 
            self.fc = nn.Sequential( 
                # LorentzLinear(self.manifold, latent_dim+1, latent_dim+1),
                LorentzMLR(self.manifold, latent_dim + 1, num_classes),
            )
        else: 
            self.fc = nn.Sequential( 
                # LorentzLinear(self.manifold, 2*self.sent_hidden_size+1, num_classes),
                LorentzMLR(self.manifold, latent_dim + 1, num_classes),
            )

    def forward(self, content, comment, subgraphs):
        
        #both content and comments modules are on
        if self.comment_module and self.content_module:

 
 
            _, content_embedding = self.content_encoder(content)
            comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
            assert not torch.isnan(content_embedding).any(), "content_embedding is nan"
            assert not torch.isnan(comment_embedding).any(), "comment_embedding is nan"

            comment_embedding = lmath.poincare_to_lorentz(comment_embedding, self.manifold.k)
            content_embedding = lmath.poincare_to_lorentz(content_embedding, self.manifold.k)
            # print(self.manifold.inner(comment_embedding, comment_embedding))
            # print(self.manifold.inner(content_embedding, content_embedding))

            # self.manifold.assert_check_point_on_manifold(comment_embedding)
            # self.manifold.assert_check_point_on_manifold(content_embedding)
            coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
            self.manifold.assert_check_point_on_manifold(coatten)
            preds = self.fc(coatten)
            return preds, As, Ac
    
        #only comment module is on
        elif self.comment_module:
            comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
            comment_embedding = lmath.poincare_to_lorentz(comment_embedding, self.manifold.k)
            preds = self.fc(comment_embedding)
            return preds

        #only content module is on
        else:
            content_embedding, _ = self.content_encoder(content)
            content_embedding = lmath.poincare_to_lorentz(content_embedding, self.manifold.k)
            preds = self.fc(content_embedding)
            return preds
    
