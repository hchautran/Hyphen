import torch
import torch.nn as nn
from geoopt import Manifold

class Model(nn.Module):

    def __init__(
        self, 
        name:str,
        embedding_matrix:torch.Tensor,  
        word_hidden_size:int, 
        sent_hidden_size:int, 
        max_sent_length:int, 
        max_word_length:int, 
        graph_hidden:int, 
        fc:nn.Module,
        content_encoder:nn.Module,
        comment_encoder:nn.Module=None,
        coattention:nn.Module=None,
        num_classes:int = 2, 
        max_sentence_count:int = 50, 
        max_comment_count:int = 10, 
        batch_size:int = 32 ,
        embedding_dim:int = 100, 
        latent_dim:int = 100, 
    ):
        super(Model, self).__init__()
        self.name = name
        self.content_encoder= comment_encoder
        self.comment_encoder = content_encoder
        self.coattention = coattention
        self.batch_size = batch_size
        self.embedding_dim = embedding_dim
        self.max_sentence_count = max_sentence_count
        self.max_comment_count = max_comment_count
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length  = max_word_length 
        self.graph_hidden = graph_hidden
        self.embedding_matrix= embedding_matrix
        self.num_classes = num_classes 
        self.latent_dim = latent_dim
        self.fc = fc 

    def forward(self, content, comment=None, comment_graph=None, subgraphs=None):
        content_embedding = self.content_encoder(content)
        if comment is  not None or comment_graph is not None:
            if comment_graph is not None and subgraphs is not None:
                comment_embedding = self.comment_encoder(comment, comment.ndata['feat'], subgraphs)
                assert not torch.isnan(content_embedding).any(), "content_embedding is nan"
                assert not torch.isnan(comment_embedding).any(), "comment_embedding is nan"
            elif comment is not None:
                comment_embedding = self.conmment_encoder(comment)
            coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
            preds = self.fc(coatten)
        else:
            preds = self.fc(content_embedding)
        return preds, As, Ac

