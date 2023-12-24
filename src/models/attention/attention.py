## import packages
import jax
import jax.numpy as jnp
import flax
import flax.linen as nn
from src.models.hippo.hippo import HiPPOLSI, HiPPOLTI
import einops
from jaxtyping import Array, Float
from typing import Optional, Tuple
import numpy as np


class ScaledDotProductAttention(nn.Module):
    """

    Attributes:
        n_head: The number of attention heads.
        d_model: The dimension of the input.
        dtype: The data type of the computation. Default is jnp.float32.
    """

    n_head: int  # number of heads the attention is split into
    d_model: int  # dimension of the input, aka n_embd or C which is the size of the embedding.
    dtype: jnp.dtype = jnp.float32  # data type of the computation (default: float32)

    def setup(self) -> None:
        # Check if d_model is divisible by n_head to ensure the input can be evenly distributed among all heads
        assert self.d_model % self.n_head == 0

        # Compute the size of each head by dividing the input dimension by the number of heads. head size, e.g. 128/4 = 32
        self.d_head = self.d_model // self.n_head

    def __call__(
        self,
        query: Float[Array, "*batch seq_len d_model"],
        key: Float[Array, "*batch seq_len d_model"],
        value: Float[Array, "*batch seq_len d_model"],
        mask: Optional[Float[Array, "*batch seq_len d_model"]] = None,
    ) -> Tuple[
        Float[Array, "*batch seq_len d_model"], Float[Array, "*batch seq_len seq_len"]
    ]:
        """
        Call method, used for calculating the forward pass for the Self Attention Module.

        Args:
            query (jnp.ndarray):
                Shape: (batch d_model)
                The query tensor.

            key (jnp.ndarray):
                Shape: (batch d_model)
                The key tensor.

            value (jnp.ndarray):
                Shape: (batch d_model)
                The value tensor.

        Returns:
            r (jnp.ndarray):
                Shape: (batch d_model)
                The projected context tensor back to the original dimension size.
        """

        key = einops.rearrange(key, "... i j -> ... j i")

        # Calculate the attention scores by taking the dot product of query and key
        score = (query @ key) / jnp.sqrt(self.d_head)

        # Masking to avoid performing attention on padding token indices.
        if mask is not None:
            assert (
                mask.shape == score.shape
            ), f"Mask shape {mask.shape} must match score shape {score.shape}"

            # Set the score for all padding token indices to a large negative value
            score = jnp.where(
                mask == 0, -9e15, score
            )  # -9e15 is a very large negative number

        # then apply softmax to get probabilities.
        attn = nn.softmax(score, axis=-1)

        # Multiply the attention scores with the value to get the context
        context = attn @ value

        return context, attn


class MultiHeadAttention(nn.Module):
    n_head: int  # number of heads the attention is split into
    d_model: int  # dimension of the input, aka n_embd or C which is the size of the embedding.
    dtype: jnp.dtype = jnp.float32  # data type of the computation (default: float32)

    def setup(self) -> None:
        # Check if d_model is divisible by n_head to ensure the input can be evenly distributed among all heads
        assert self.d_model % self.n_head == 0

        # Compute the size of each head by dividing the input dimension by the number of heads. head size, e.g. 128/4 = 32
        self.d_head = self.d_model // self.n_head

        # Create dense layers for key, query, and value with the dimension size of each head.

        self.key = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            name=f"key_layer",
        )
        self.query = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            name=f"query_layer",
        )
        self.value = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            name=f"value_layer",
        )

        # Create a dense layer for projecting the output back to the original dimension size.
        self.proj = nn.Dense(
            self.d_model,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            name=f"proj_layer",
        )

        self.attention = ScaledDotProductAttention(
            n_head=self.n_head, d_model=self.d_model
        )

    def __call__(
        self,
        query: Float[Array, "*batch seq_len d_model"],
        key: Float[Array, "*batch seq_len d_model"],
        value: Float[Array, "*batch seq_len d_model"],
        mask: Optional[Float[Array, "*batch seq_len d_model"]] = None,
    ) -> Tuple[
        Float[Array, "*batch seq_len d_model"], Float[Array, "*batch seq_len seq_len"]
    ]:

        q = self.query(query)
        k = self.key(key)
        v = self.value(value)

        q = einops.rearrange(
            q,
            "... seq_len (n_head d_head) -> ... seq_len n_head d_head",
            n_head=self.n_head,
            d_head=self.d_head,
        )
        k = einops.rearrange(
            k,
            "... seq_len (n_head d_head) -> ... seq_len n_head d_head",
            n_head=self.n_head,
            d_head=self.d_head,
        )
        v = einops.rearrange(
            v,
            "... seq_len (n_head d_head) -> ... seq_len n_head d_head",
            n_head=self.n_head,
            d_head=self.d_head,
        )

        context, attn = jax.vmap(self.attention, in_axes=(1, 1, 1, None))(q, k, v, mask)

        context = einops.rearrange(
            context,
            "... n_head seq_len d_head -> ... seq_len (n_head d_head)",
            n_head=self.n_head,
            d_head=self.d_head,
        )

        # Project the context back to the original dimension size using the projection layer
        out = self.proj(context)

        return out, attn


class MultiHiPPOLTIAttention(nn.Module):
    n_head: int  # number of heads the attention is split into
    d_model: int  # dimension of the input, aka n_embd or C which is the size of the embedding.
    step: float  # step size for the GBT
    lambda_n: float = 1.0  # lambda_n for the LegT
    alpha: float = 2.0  # alpha for the GBT,
    measure: str = "legs"  # measure for type of the polynomial,
    basis: float = 1.0  # basis for the polynomial
    unroll: bool = False  # unroll the loop for the output
    recon: bool = False
    dtype: jnp.dtype = jnp.float32  # data type of the computation (default: float32)

    def setup(self) -> None:
        # Check if d_model is divisible by n_head to ensure the input can be evenly distributed among all heads
        assert self.d_model % self.n_head == 0

        # Compute the size of each head by dividing the input dimension by the number of heads. head size, e.g. 128/4 = 32
        self.d_head = self.d_model // self.n_head

        self.out_dim = 1000  # int(T / self.step)

        # Create dense layers for key, query, and value with the dimension size of each head.

        self.query = HiPPOLTI(
            N=128,
            step_size=self.step,
            lambda_n=self.lambda_n,
            alpha=0.0,
            beta=1.0,
            GBT_alpha=self.alpha,
            measure=self.measure,
            basis_size=self.basis,
            dtype=jnp.float32,
            unroll=False,
            recon=self.recon,
        )

        self.key = HiPPOLTI(
            N=128,
            step_size=self.step,
            lambda_n=self.lambda_n,
            alpha=0.0,
            beta=1.0,
            GBT_alpha=self.alpha,
            measure=self.measure,
            basis_size=self.basis,
            dtype=jnp.float32,
            unroll=False,
            recon=self.recon,
        )

        self.value = HiPPOLTI(
            N=128,
            step_size=self.step,
            lambda_n=self.lambda_n,
            alpha=0.0,
            beta=1.0,
            GBT_alpha=self.alpha,
            measure=self.measure,
            basis_size=self.basis,
            dtype=jnp.float32,
            unroll=False,
            recon=self.recon,
        )

        # Create a dense layer for projecting the output back to the original dimension size.
        self.proj = nn.Dense(
            self.out_dim,
            kernel_init=nn.initializers.xavier_uniform(),  # Weights with Xavier uniform init
            bias_init=nn.initializers.zeros,  # Bias init with zeros
            name=f"proj_layer",
        )

        self.attention = ScaledDotProductAttention(
            n_head=self.n_head, d_model=self.d_model
        )

    def __call__(
        self,
        query: Float[Array, "*batch seq_len d_model"],
        key: Float[Array, "*batch seq_len d_model"],
        value: Float[Array, "*batch seq_len d_model"],
        mask: Optional[Float[Array, "*batch seq_len d_model"]] = None,
    ) -> Tuple[
        Float[Array, "*batch seq_len d_model"], Float[Array, "*batch seq_len seq_len"]
    ]:

        query = einops.rearrange(query, "seq_len d_model -> 1 seq_len d_model")
        key = einops.rearrange(key, "seq_len d_model -> 1 seq_len d_model")
        value = einops.rearrange(value, "seq_len d_model -> 1 seq_len d_model")

        q, _ = self.query(query)
        k, _ = self.key(key)
        v, _ = self.value(value)

        q = einops.rearrange(q, "seq_len 1 N -> seq_len N")
        k = einops.rearrange(k, "seq_len 1 N -> seq_len N")
        v = einops.rearrange(v, "seq_len 1 N -> seq_len N")

        context, attn = self.attention(q, k, v, mask)
        context = einops.rearrange(context, "N 1 -> 1 N")

        # Project the context back to the original dimension size using the projection layer
        out = self.proj(context)
        out = einops.rearrange(out, "1 N -> N 1")

        return out, attn
