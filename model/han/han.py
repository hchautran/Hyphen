import torch
import torch.nn as nn
import torch.nn.functional as F
import csv
import sys
csv.field_size_limit(sys.maxsize)
import sklearn
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings#ignoring the undefinedmetric warnings -- incase of precision having zero division
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) 
from ..coattention.poincare import CoAttention as PoincareCoAttn
from ..coattention.euclidean import CoAttention as EuclidCoAttn
from ..coattention.lorentz import CoAttention as LorentzCoAttn
from hyptorch.lorentz.manifold import CustomLorentz 
from ..utils.utils import matrix_mul, element_wise_mul
from typing import Union
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers import LorentzMLR 
from hyptorch.geoopt import PoincareBall, Euclidean


class HanEnc(nn.Module):
    def __init__(
        self, 
        manifold:Union[PoincareBall, CustomLorentz, Euclidean],
        word_hidden_size, 
        sent_hidden_size, 
        batch_size, 
        embedding_matrix, 
        device, 
    ):
        super(HanEnc, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.manifold = manifold

        self.word_att_net = WordAttNet(embedding_matrix, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size)
        
        self._init_hidden_state()
        
    def _init_hidden_state(self, last_batch_size=None):
        if last_batch_size:
            batch_size = last_batch_size
        else:
            batch_size = self.batch_size
        self.word_hidden_state = torch.zeros(2, batch_size, self.word_hidden_size)
        self.sent_hidden_state = torch.zeros(2, batch_size, self.sent_hidden_size)
        self.word_hidden_state = self.word_hidden_state.to(self.device)
        self.sent_hidden_state = self.sent_hidden_state.to(self.device)


    def forward(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, _= self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        _, x= self.sent_att_net(output, self.sent_hidden_state)

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

    def step(self, input):
        output_list = []
        input = input.permute(1, 0, 2)
        for i in input:
            output, _ = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)

        _, x= self.sent_att_net(output, self.sent_hidden_state)
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



class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50):
        super(SentAttNet, self).__init__()
        self.sent_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 2 * sent_hidden_size))
        self.sent_bias = nn.Parameter(torch.Tensor(1, 2 * sent_hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * sent_hidden_size, 1))
        self.gru = nn.GRU(2 * word_hidden_size, sent_hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.sent_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        f_output, _ = self.gru(input, hidden_state)
        output = matrix_mul(f_output, self.sent_weight, self.sent_bias)
        output = matrix_mul(output, self.context_weight).permute(1, 0)
        output = F.softmax(output, dim = -1)
        output = element_wise_mul(f_output, output.permute(1, 0)).squeeze(0)
        return output, f_output.permute(1, 0, 2) #return none curvature



class WordAttNet(nn.Module):
    def __init__(self, embedding_matrix, hidden_size=50):
        super(WordAttNet, self).__init__()
        self.word_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.word_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.context_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.lookup = self.create_embeddeding_layer(embedding_matrix)
        self.gru = nn.GRU(self.embedding_dim, hidden_size, bidirectional=True)
        self._create_weights(mean=0.0, std=0.05)

    def _create_weights(self, mean=0.0, std=0.05):
        self.word_weight.data.normal_(mean, std)
        self.context_weight.data.normal_(mean, std)

    def forward(self, input, hidden_state):
        output = self.lookup(input)
        f_output, h_output = self.gru(output.float(), hidden_state)  # feature output and hidden state output
        output = matrix_mul(f_output, self.word_weight, self.word_bias)
        output = matrix_mul(output, self.context_weight).permute(1,0)
        output = F.softmax(output, dim = -1)
        output = element_wise_mul(f_output,output.permute(1,0))
        return output, h_output
        
    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer


class Han(nn.Module):

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

        super(Han,self).__init__()
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
        self.content_encoder= HanEnc(
            manifold=self.manifold, 
            word_hidden_size=word_hidden_size,
            sent_hidden_size=sent_hidden_size,
            embedding_matrix=embedding_matrix,
            device=device,
            batch_size=batch_size
        )
        print('building HypComEnc')
        self.comment_encoder= HanEnc(
            manifold=self.manifold, 
            word_hidden_size=word_hidden_size,
            sent_hidden_size=sent_hidden_size,
            embedding_matrix=embedding_matrix,
            device=device,
            batch_size=batch_size
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

    def step(self, content:torch.Tensor=None, comment:torch.Tensor=None):
        content_embedding = self.content_encoder(content)
        comment_embedding = self.content_encoder(comment)
        coatten, As, Ac = self.coattention(content_embedding, comment_embedding)
        preds = self.fc(coatten)
        return preds, As, Ac