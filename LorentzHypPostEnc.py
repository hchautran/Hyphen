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
import warnings
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning) 

from geoopt import ManifoldParameter
from hyptorch.geoopt.manifolds import PoincareBall, Euclidean 
from utils.nets import MobiusGRU
from utils.nets import MobiusLinear
from utils.nets import MobiusDist2Hyperplane
from utils.utils import matrix_mul, element_wise_mul
from transformers import PerceiverLayer

eps = 1e-7

class HypPostEnc(nn.Module):
    def __init__(self, word_hidden_size, sent_hidden_size, batch_size, num_classes, embedding_matrix, max_sent_length, max_word_length, device, manifold,
    content_curvature):
        super(HypPostEnc, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.word_hidden_size = word_hidden_size
        self.sent_hidden_size = sent_hidden_size
        self.max_sent_length = max_sent_length
        self.max_word_length = max_word_length
        self.manifold = manifold
        self.content_curvature = content_curvature

        if isinstance(self.manifold, Euclidean):
            self.word_att_net = WordAttNet(embedding_matrix, word_hidden_size)
            self.sent_att_net = SentAttNet(sent_hidden_size, word_hidden_size, num_classes)

        else:
            self.word_att_net = H_WordAttNet(embedding_matrix, word_hidden_size)
            self.sent_att_net = H_SentAttNet(sent_hidden_size, word_hidden_size, num_classes, self.content_curvature)
        
        
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

def E2Lorentz(input):
    """Function to convert fromm Euclidean space to the Lorentz model"""
    rr = torch.norm(input, p=2, dim=2)
    dd = input.permute(2,0,1) / rr
    cosh_r = torch.cosh(rr)
    sinh_r = torch.sinh(rr)
    output = torch.cat(((dd * sinh_r).permute(1, 2, 0), cosh_r.unsqueeze(0).permute(1, 2, 0)), dim=2)
    return output

def P2Lorentz(input):
    """Function to convert fromm Poincare model to the Lorentz model"""
    rr = torch.norm(input, p=2, dim=2)
    output = torch.cat((2*input, (1+rr**2).unsqueeze(2)),dim=2).permute(2,0,1)/(1-rr**2+eps)
    return output.permute(1,2,0)

def L2Klein(input):
    """Function to convert fromm Lorentz model to the Klein model"""
    dump = input[:, :, -1]
    dump = torch.clamp(dump, eps, 1.0e+16)
    return (input[:, :, :-1].permute(2, 0, 1)/dump).permute(1, 2, 0)

def arcosh(x):
    c0 = torch.log(x)
    c1 = torch.log1p(torch.sqrt(x * x - 1 + eps) / x)
    return c0 + c1

def disLorentz(x, y):
    m = x * y
    prod_minus = -m[:, :, :-1].sum(dim=2) + m[:, :, -1]
    output = torch.clamp(prod_minus, 1.0 + eps, 1.0e+16)
    return arcosh(output)


class PerceiverLayer(nn.Module):
    def __init__(
        self,
        config,
        is_cross_attention=False,
        qk_channels=None,
        v_channels=None,
        num_heads=1,
        q_dim=None,
        kv_dim=None,
        widening_factor=4,
        use_query_residual=True,
    ):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = PerceiverAttention(
            config,
            is_cross_attention=is_cross_attention,
            qk_channels=qk_channels,
            v_channels=v_channels,
            num_heads=num_heads,
            q_dim=q_dim,
            kv_dim=kv_dim,
            use_query_residual=use_query_residual,
        )
        self.layernorm = nn.LayerNorm(q_dim)
        self.mlp = PerceiverMLP(config, input_size=q_dim, widening_factor=widening_factor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        inputs: Optional[torch.FloatTensor] = None,
        inputs_mask: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor]:
        attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            inputs,
            inputs_mask,
            output_attentions,
        )
        attention_output = attention_outputs[0]

        outputs = attention_outputs[1:]  # add attentions if we output attention weights

        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )

        layer_output = layer_output + attention_output  # residual connection

        outputs = (layer_output,) + outputs

        return outputs

    def feed_forward_chunk(self, attention_output):
        layer_output = self.layernorm(attention_output)
        layer_output = self.mlp(layer_output)
        return layer_output

