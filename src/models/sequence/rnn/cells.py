from flax import linen as nn
import jax.numpy as jnp


class RNNCell(nn.Module):
    input_size: int
    hidden_size: int
    bias: bool = True
    nonlinearity: str = "tanh"

    def setup(self):
        if self.nonlinearity not in ["tanh", "relu"]:
            raise ValueError("Invalid nonlinearity selected for RNN.")

        self.W_hh = nn.Dense(self.hidden_size, use_bias=self.bias)
        self.W_xh = nn.Dense(self.hidden_size, use_bias=self.bias)

    def __call__(self, carry, input):
        ht_1, _ = carry
        print(f"ht_1 shape: {ht_1.shape}")
        print(f"input shape: {input.shape}")
        # x = jnp.concatenate([ht_1, input], axis=0)
        W_hh = self.W_hh(ht_1)
        W_xh = self.W_xh(input)
        print(f"W_hh shape: {W_hh.shape}")
        print(f"W_xh shape: {W_xh.shape}")

        if self.nonlinearity == "tanh":
            h_t = nn.tanh(
                W_hh + W_xh
            )  # H_{t} = tanh(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})
        else:
            h_t = nn.relu(
                W_hh + W_xh
            )  # H_{t} = tanh(H_{t-1} @ W_{hh}) + (x_{t} @ W_{xh})

        return (h_t, h_t), h_t
