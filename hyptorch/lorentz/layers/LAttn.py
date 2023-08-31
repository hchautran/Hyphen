import torch
import torch.nn as nn
from typing import Optional, Tuple
from hyptorch.lorentz.manifold import CustomLorentz
from hyptorch.lorentz.layers.LFC import LorentzLinear



class CrossAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, manifold:CustomLorentz, config):
        super().__init__()
        self.config = config
        self.manifold = manifold
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config.attention_dropout
        self.bias = nn.Parameter(torch.zeros(()))

        self.k_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1, normalize=True)
        self.v_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1, normalize=True)
        self.q_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1, normalize=True)
        self.out_proj = LorentzLinear(manifold, self.embed_dim + 1, self.embed_dim + 1, normalize=True)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        space = tensor.narrow(-1, 1, tensor.shape[-1] - 1)
        space = space.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()
        tensor = self.manifold.add_time(space) 
        return tensor 

    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = x.size()

        # get query proj
        query_states = self.q_proj(x) 
        key_states = self._shape(self.k_proj(y), -1, bsz)
        value_states = self._shape(self.v_proj(y), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim + 1)

        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)
        
        self.manifold.assert_check_point_on_manifold(query_states)
        self.manifold.assert_check_point_on_manifold(key_states)
        self.manifold.assert_check_point_on_manifold(value_states)

        src_len = key_states.size(1)
        attn_weights = (
            2 + 2 * self.manifold.matmul(query_states, key_states.transpose(1, 2))
        ) * self.scale + self.bias

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None


        attn_output = self.manifold.projx(self.manifold.centroid(value_states, attn_weights))
        self.manifold.assert_check_point_on_manifold(attn_output)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim + 1):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim + 1)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim + 1)
        attn_output = attn_output.transpose(1, 2)
        space = attn_output.narrow(-1, 1, attn_output.shape[-1] - 1)
        attn_output = space.reshape(bsz, tgt_len, embed_dim - 1)

        attn_output = self.manifold.add_time(attn_output)
        self.manifold.assert_check_point_on_manifold(attn_output)

        attn_output = self.out_proj(attn_output)
        self.manifold.assert_check_point_on_manifold(attn_output)

        return attn_output, attn_weights_reshaped

