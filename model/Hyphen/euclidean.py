import torch
import torch.nn as nn
import torch.nn as nn

from ..coattention.euclidean import CoAttention
from ..comenc.euclidean_gcn import ComEnc 
from ..postenc.euclidean_han import PostEnc 
from ..utils.layers.hyp_layers import *

class Hyphen(nn.Module):

    def __init__(
        self, embedding_matrix,  word_hidden_size, sent_hidden_size, max_sent_length, max_word_length, device, graph_hidden, num_classes = 2, max_sentence_count = 50 , max_comment_count = 10, batch_size = 32 ,embedding_dim = 100, latent_dim = 100, graph_glove_dim = 100, manifold = "hyper",
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
        self.comment_module = comment_module
        self.content_module = content_module 
        print('building HypPostEnc')
        self.content_encoder= ComEnc(self.word_hidden_size, self.sent_hidden_size, batch_size, num_classes, embedding_matrix, self.max_sent_length, self.max_word_length, self.device, manifold = self.manifold, content_curvature = self.content_curvature)
        print('building HypComEnc')
        self.comment_encoder = PostEnc(self.graph_glove_dim, self.graph_hidden, num_classes, self.max_comment_count, device= self.device, manifold = self.manifold, content_module = self.content_module, comment_curvature = self.comment_curvature)
        print('building CoAttention')
        self.coattention = CoAttention(device, latent_dim, manifold = self.manifold,  comment_curvature = self.comment_curvature, content_curvature = self.content_curvature, combined_curvature = self.combined_curvature, fourier = self.fourier)
        
        if self.comment_module and self.content_module: self.fc = nn.Linear(2*latent_dim, num_classes)
        elif self.comment_module: self.fc = nn.Linear(latent_dim, num_classes)
        else: self.fc = nn.Linear(2*self.sent_hidden_size, num_classes)

    def forward(self, content, comment, subgraphs):
        
        if self.comment_module and self.content_module:
            _, content_embedding = self.content_encoder(content)
            comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
            coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
            preds = self.fc(coatten)
            return preds, As, Ac
        elif self.comment_module:
            comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
            preds = self.fc(comment_embedding)
            return preds
        else:
            content_embedding, _ = self.content_encoder(content)
            preds = self.fc(content_embedding)
            return preds
    
