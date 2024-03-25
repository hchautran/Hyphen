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
from ..utils.utils import matrix_mul, element_wise_mul


class PostEnc(nn.Module):
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
    ):
        super(PostEnc, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length

        self.word_att_net = WordAttNet(embedding_matrix, word_hidden_size)
        self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)
        
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
            output, self.word_hidden_state = self.word_att_net(i.permute(1, 0), self.word_hidden_state)
            output_list.append(output)
        output = torch.cat(output_list, 0)
        output, h_output = self.sent_att_net(output, self.sent_hidden_state)
        return output, h_output


class SentAttNet(nn.Module):
    def __init__(self, sent_hidden_size=50, word_hidden_size=50, num_classes=14):
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
        f_output, h_output = self.gru(input, hidden_state)
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
