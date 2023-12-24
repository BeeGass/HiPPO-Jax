## import packages
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from src.models.hippo.hippo import HiPPOLSI, HiPPOLTI
from src.models.attention.attention import (
    ScaledDotProductAttention,
    MultiHeadAttention,
    MultiHiPPOLTIAttention,
)
import einops
from jaxtyping import Array, Float
from typing import Optional, Tuple
import numpy as np


class TransformerBlock(nn.Module):
    d_model: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    n_head: int  # number of heads the attention is split into
    ffn_expan: int  # expansion factor for the feedforward layer
    _dropout: float  # dropout probability
    dtype: jnp.dtype = jnp.float32  # data type of the computation (default: float32)

    def setup(self):
        # Attention layer
        self.attention = MultiHeadAttention(n_head=self.n_head, d_model=self.d_model)

        # Two-layer MLP
        self.ffn = [
            nn.Dense(self.ffn_expan * self.d_model),
            nn.Dropout(rate=self._dropout),
            nn.relu,
            nn.Dense(self.d_model),
        ]
        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self._dropout)

    def __call__(
        self,
        query: Float[Array, "*batch seq_len d_model"],
        key: Float[Array, "*batch seq_len d_model"],
        value: Float[Array, "*batch seq_len d_model"],
        mask: Optional[Float[Array, "*batch seq_len d_model"]] = None,
        train: bool = True,
    ) -> Float[Array, "*batch seq_len d_model"]:

        # Attention part
        proj_context, attn = jax.vmap(self.attention, in_axes=(0, 0, 0, 0))(
            self.norm1(query), self.norm1(key), self.norm1(value), mask
        )
        x = query + self.dropout(proj_context, deterministic=not train)

        # MLP part
        linear_out = self.norm2(x)
        for layer in self.ffn:
            if not isinstance(layer, nn.Dropout):
                linear_out = layer(linear_out)
            else:
                linear_out = layer(linear_out, deterministic=not train)

        x = x + self.dropout(linear_out, deterministic=not train)

        return x


class TransformerDecoderBlock(nn.Module):
    d_model: int  # Input dimension is needed here since it is equal to the output dimension (residual connection)
    n_head: int  # number of heads the attention is split into
    ffn_expan: int  # expansion factor for the feedforward layer
    dropout: float  # dropout probability

    def setup(self):
        # Attention layer
        self.attention = MultiHeadAttention(n_head=self.n_head, d_model=self.d_model)
        self.transformer_block = TransformerBlock(
            d_model=self.d_model,
            n_head=self.n_head,
            ffn_expan=self.ffn_expan,
            _dropout=self.dropout,
        )

        # Layers to apply in between the main layers
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.dropout = nn.Dropout(rate=self.dropout)

    def __call__(
        self,
        x: Float[Array, "*batch d_model"],
        key: Float[Array, "*batch d_model"],
        value: Float[Array, "*batch d_model"],
        mask: Optional[Float[Array, "*batch d_model"]] = None,
        trg_mask: Optional[Float[Array, "*batch d_model"]] = None,
        train: bool = True,
    ) -> Float[Array, "*batch d_model"]:
        # Masked Attention part
        mask_proj_context, mask_attn = jax.vmap(self.attention, in_axes=(0, 0, 0, 0))(
            self.norm1(x), self.norm1(key), self.norm1(value), trg_mask
        )
        query = x + self.dropout(mask_proj_context, deterministic=not train)

        # Encoder Attention part
        out = self.transformer_block(
            self.norm2(query), self.norm2(key), self.norm2(value), mask=mask
        )

        return out
