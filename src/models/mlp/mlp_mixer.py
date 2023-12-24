## import packages
import math
from typing import Any, Callable, List, Optional, Tuple, Union

import einops
import jax
import jax.numpy as jnp
from flax import linen as nn
from jaxtyping import Array, Float, Int
from scipy import special as ss

from src.models.hippo.transition import (
    LegSInitializer,
    LegTInitializer,
    LMUInitializer,
    LagTInitializer,
    FRUInitializer,
    FouTInitializer,
    FouDInitializer,
    ChebTInitializer,
    legs_initializer,
    legt_initializer,
    lmu_initializer,
    lagt_initializer,
    fru_initializer,
    fout_initializer,
    foud_initializer,
    chebt_initializer,
)
from src.models.hippo.hippo import HiPPOLSI, HiPPOLTI
from src.models.model import Model


class MLPBlock(nn.Module):
    mlp_dim: int

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch channels"]
    ) -> Float[Array, "batch channels"]:
        y = nn.Dense(self.mlp_dim)(x)
        y = nn.gelu(y)
        return nn.Dense(x.shape[-1])(y)


class MixerBlock(nn.Module):
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch channels"]
    ) -> Float[Array, "batch channels"]:
        y = nn.LayerNorm()(x)
        y = jnp.swapaxes(y, 1, 2)
        y = MLPBlock(self.tokens_mlp_dim, name=" token_mixing ")(y)
        y = jnp.swapaxes(y, 1, 2)
        x = x + y
        y = nn.LayerNorm()(x)
        return x + MLPBlock(self.channels_mlp_dim, name=" channel_mixing ")(y)


class MLPMixer(nn.Module):
    num_classes: int
    num_blocks: int
    patch_size: int
    hidden_dim: int
    tokens_mlp_dim: int
    channels_mlp_dim: int

    @nn.compact
    def __call__(
        self, x: Float[Array, "batch channels height width"]
    ) -> Int[Array, "batch"]:
        s = self.patch_size
        x = nn.Conv(self.hidden_dim, (s, s), strides=(s, s), name="stem ")(x)
        x = einops.rearrange(x, "n h w c -> n (h w) c")
        for _ in range(self.num_blocks):
            x = MixerBlock(self.tokens_mlp_dim, self.channels_mlp_dim)(x)
        x = nn.LayerNorm(name=" pre_head_layer_norm ")(x)
        x = jnp.mean(x, axis=1)
        return nn.Dense(
            self.num_classes, name="head ", kernel_init=nn.initializers.zeros
        )(x)
