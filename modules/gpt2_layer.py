from torch import nn

import torch.nn.functional as F

from modules.attention import CausalSelfAttention

class GPT2Layer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # Multi-head attention.
    self.self_attention = CausalSelfAttention(config)
    # Add-norm for multi-head attention.
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # Feed forward.
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # Add-norm for feed forward.
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  def add(self, input, output, dense_layer, dropout):
    """
    Apply dense projection and dropout to output, then add residual connection.
    Layer norm is NOT applied here.
    """
    return input + dropout(dense_layer(output))

  def forward(self, hidden_states, attention_mask):
    """
    Pre-norm GPT-2 transformer layer:
      LayerNorm -> MHA -> residual
      LayerNorm -> FFN -> residual
    """
    # Self-attention sub-layer (pre-norm)
    attn_out = self.self_attention(self.attention_layer_norm(hidden_states), attention_mask)
    hidden_states = self.add(hidden_states, attn_out, self.attention_dense, self.attention_dropout)

    # Feed-forward sub-layer (pre-norm)
    interm = self.interm_af(self.interm_dense(self.out_layer_norm(hidden_states)))
    hidden_states = self.add(hidden_states, interm, self.out_dense, self.out_dropout)

    return hidden_states

