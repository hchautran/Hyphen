import torch
import torch.nn as nn
from ..coattention.lorentz import CoAttention
from ..comenc.lorentz_gcn import ComEnc 
from ..postenc.lorentz_han import PostEnc 
from ..utils.layers.hyp_layers import *
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers import LorentzMLR 

class Hyphen(nn.Module):

    def __init__(
        self, 
        manifold:CustomLorentz,
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
        content_module =True, 
        comment_module = True, 
        fourier = False
    ):

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
        self.manifold = CustomLorentz(k=1.0) 
        self.poincare = PoincareBall(c=self.manifold.k.item())
        self.comment_module = comment_module
        self.content_module = content_module 
        print('building HypPostEnc')
        self.content_encoder= PostEnc(
            word_hidden_size=self.word_hidden_size, 
            sent_hidden_size=self.sent_hidden_size, 
            batch_size=batch_size, 
            num_classes=num_classes, 
            embedding_matrix=embedding_matrix, 
            max_sent_length=self.max_sent_length, 
            max_word_length=self.max_word_length, 
            device=self.device, 
            manifold=self.poincare, 
        )
        print('building HypComEnc')
        self.comment_encoder = ComEnc(
            in_dim=self.graph_glove_dim, 
            hidden_dim=self.graph_hidden, 
            max_comment_count=self.max_comment_count, 
            device=self.device, 
            manifold=self.poincare, 
            content_module=self.content_module, 
            comment_curvature=self.manifold.k.item()
        )
        print('building CoAttention')
        self.coattention = CoAttention(
            embedding_dim=latent_dim, 
            latent_dim=latent_dim, 
            manifold=self.manifold,  
            fourier=self.fourier
        )
        self.fc = nn.Sequential(
            # nn.Linear(2*latent_dim, num_classes),
            LorentzMLR(manifold=self.manifold, num_features=2*latent_dim+1,num_classes=num_classes)
                
        )

    def forward(self, content, comment, subgraphs):
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
    
    
