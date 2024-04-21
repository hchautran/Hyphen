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
from utils.manifolds import Euclidean, PoincareBall
from utils.nets import MobiusGRU
from utils.nets import MobiusLinear
from utils.nets import MobiusDist2Hyperplane
from utils.utils import matrix_mul, element_wise_mul
from transformers import DistilBertConfig, DistilBertModel

eps = 1e-7
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) 

class HyphatPost(nn.Module):
    def __init__(
        self, 
        word_hidden_size, 
        sent_hidden_size, 
        batch_size, 
        num_classes, 
        embedding_matrix, 
        max_sent_length, 
        max_word_length, 
        device, 
        manifold,
        content_curvature
    ):
        super(HyphatPost, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.word_config = DistilBertConfig(
            dim=100, 
            hidden_dim=2*word_hidden_size, 
            n_layers=3,
            n_heads=2
        )
        self.sent_config = DistilBertConfig(
            dim=100, 
            hidden_dim=2*sent_hidden_size, 
            n_layers=1,
            n_heads=2
        )
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.manifold = manifold
        self.content_curvature = content_curvature

        self.word_att_net = WordBert(config=self.word_config, embedding_matrix=embedding_matrix)
        self.sent_att_net = SentBert(config=self.sent_config)

        
    def forward(self, input):
        output_list = []
        # input = input.permute(1, 0, 2)
        for i in input:
            output = self.word_att_net(i)
            # print(output)
            output_list.append(output[0][:,0,:])
            # print(output[0][:,0,:].shape)
        output = torch.stack(output_list, dim=0)
        # print(output.shape)
        output = self.sent_att_net(output)[0]
        output = self.manifold.expmap0(output, c=1.0)
        return output

        
class WordBert(nn.Module):
    def __init__(self, config:DistilBertConfig, embedding_matrix:torch.Tensor):
        super(WordBert, self).__init__()

        self.model = DistilBertModel(config=config)
        self.lookup = self.create_embeddeding_layer(embedding_matrix)

    def create_embeddeding_layer(self, weights_matrix, trainable=False):
        self.num_embeddings, self.embedding_dim = weights_matrix.shape
        weights_matrix = torch.from_numpy(weights_matrix)
        emb_layer = nn.Embedding(self.num_embeddings, self.embedding_dim)
        emb_layer.load_state_dict({'weight': weights_matrix})
        emb_layer.weight.requires_grad = trainable
        return emb_layer

    def forward(self, input):
        hidden_state = self.lookup(input)
        # print(hidden_state.shape)
        output = self.model(inputs_embeds=hidden_state) 
        return output

        
        
class SentBert(nn.Module):
    def __init__(self, config:DistilBertConfig):
        super(SentBert, self).__init__()
        self.model = DistilBertModel(config=config)

    def forward(self, hidden_state):
        output = self.model(inputs_embeds=hidden_state)
        return output
