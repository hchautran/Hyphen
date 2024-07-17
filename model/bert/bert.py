import torch
import torch.nn as nn
from typing import Union
from hyptorch.geoopt import PoincareBall
from transformers import BertModel,BertConfig 
from hyptorch.lorentz.manifold import CustomLorentz 
from hyptorch.geoopt import Euclidean 
from ..coattention.poincare import CoAttention as PoincareCoAttn
from ..coattention.euclidean import CoAttention as EuclidCoAttn
from ..coattention.lorentz import CoAttention as LorentzCoAttn
from hyptorch.lorentz.layers import LorentzMLR 
import torch.nn.functional as F


class BertEnc(nn.Module):
    def __init__(
        self, 
        manifold:Union[PoincareBall, CustomLorentz, Euclidean],
        word_hidden_size, 
        sent_hidden_size, 
        embedding_matrix, 
        factor=2,
    ):
        super(BertEnc, self).__init__()
        self.word_config = BertConfig(
            max_position_embeddings=30,
            hidden_size=word_hidden_size,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=word_hidden_size*factor
        )
        self.sent_config = BertConfig( 
            max_position_embeddings=2048,
            hidden_size=sent_hidden_size,
            num_hidden_layers=1,
            num_attention_heads=1,
            intermediate_size=sent_hidden_size*factor
        )

        self.manifold = manifold
        self.word_bert= BertModel(self.word_config) 
        self.sent_ssm= BertModel(self.sent_config) 
        self.lookup = self.create_embeddeding_layer(embedding_matrix)
    
    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer



    def forward(self, input:torch.Tensor):
        output_list = []
        # input = input.permute(1, 0, 2)
        for x in input:
            x = self.lookup(x)
            x = self.word_bert(inputs_embeds=x).pooler_output
            output_list.append(x)

        output = torch.stack(output_list, dim=0)
        x = self.sent_ssm(inputs_embeds=output).last_hidden_state
        if isinstance(self.manifold, PoincareBall):
            clip_r = 2.0 
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), clip_r/ x_norm)
            x = x * fac
            output = self.manifold.expmap0(x)
        elif isinstance(self.manifold, CustomLorentz):
            clip_r = 2.0 
            x_norm = torch.norm(x, dim=-1, keepdim=True) + 1e-5
            fac = torch.minimum(torch.ones_like(x_norm), clip_r/ x_norm)
            x = x * fac
            x = F.pad(x, (1,0), "constant", 0)
            output = self.manifold.expmap0(x)

        return output 


class HBert(nn.Module):

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

        super(HBert,self).__init__()
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
        self.content_encoder= BertEnc(
            manifold=self.manifold, 
            word_hidden_size=word_hidden_size*2,
            sent_hidden_size=sent_hidden_size*2,
            embedding_matrix=embedding_matrix,
        )
        print('building HypComEnc')
        self.comment_encoder= BertEnc(
            manifold=self.manifold, 
            word_hidden_size=word_hidden_size*2,
            sent_hidden_size=sent_hidden_size*2,
            embedding_matrix=embedding_matrix,
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