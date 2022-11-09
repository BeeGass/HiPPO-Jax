import jax
from flax import linen as nn
from flax.linen.initializers import zeros
import jax.numpy as jnp
from cells import RNNCell
from typing import Any, Callable, Sequence, Optional, Tuple, Union
from dataclasses import field


class SimpleRNN(nn.Module):
    input_size: int
    hidden_size: int
    num_layers: int
    output_size: int
    bias: bool = True
    activation: str = "tanh"
    # rnn_cells: Sequence[nn.Module] = field(default_factory=list)

    def setup(self):
        rnn_cells = []
        if self.activation == "tanh":
            for l in range(self.num_layers):
                if l == 0:
                    input_size = self.input_size
                else:
                    input_size = self.hidden_size

                rnn_cells.append(
                    RNNCell(
                        input_size=input_size,
                        hidden_size=self.hidden_size,
                        bias=self.bias,
                        nonlinearity="tanh",
                    )
                )

        elif self.activation == "relu":
            for l in range(self.num_layers):
                if l == 0:
                    input_size = self.input_size
                else:
                    input_size = self.hidden_size

                rnn_cells.append(
                    RNNCell(
                        input_size=self.input_size,
                        hidden_size=self.hidden_size,
                        bias=self.bias,
                        nonlinearity="relu",
                    )
                )
        else:
            raise ValueError("Invalid activation.")

        self.rnn_cells = rnn_cells

        self.fc = nn.Dense(self.output_size)

    def __call__(self, carry, input):

        # Input of shape (batch_size, seqence length, input_size)
        #
        # Output of shape (batch_size, output_size)
        h_t, c_t = carry
        print(f"input shape: {input.shape}")
        print(f"h_t shape: {h_t.shape}")
        out = []
        hidden = []
        for layer in range(self.num_layers):
            # new_carry = (jnp.expand_dims(h_t, axis=-1), jnp.expand_dims(c_t, axis=-1))
            hidden.append(carry)

        for t in range(input.shape[1]):
            for layer in range(self.num_layers):
                if layer == 0:
                    hidden_l = self.rnn_cells[layer](hidden[layer], input[:, :, t])

                else:
                    new_carry = hidden[layer - 1]
                    # TODO: there is an issue where lists are becoming tuples preventing this logic from being possible
                    print(f"new_carry: {new_carry}")
                    new_input, _ = new_carry
                    print(f"hidden[layer - 1]: {hidden[layer - 1]}")
                    print(f"new_input: {new_input}")
                    hidden_l = self.rnn_cells[layer](hidden[layer], new_input)

                hidden[layer] = hidden_l

            out.append(hidden_l)

        # Take only last time step. Modify for seq to seq
        out = jnp.expand_dims(out, axis=-1)

        return self.fc(out)

    @staticmethod
    def initialize_carry(rng, batch_dims, size, init_fn=zeros):
        """Initialize the RNN cell carry.

        Args:
        rng: random number generator passed to the init_fn.
        batch_dims: a tuple providing the shape of the batch dimensions.
        size: the size or number of features of the memory.
        init_fn: initializer function for the carry.
        Returns:
        An initialized carry for the given RNN cell.
        """
        key1, key2 = jax.random.split(rng)
        mem_shape = batch_dims + (size,)
        return init_fn(key1, mem_shape), init_fn(key2, mem_shape)


def test():
    seed = 1701
    key = jax.random.PRNGKey(seed)

    num_copies = 4
    rng, key, subkey, subsubkey = jax.random.split(key, num=num_copies)

    hidden_size = 256

    # batch size, sequence length, input size
    batch_size = 64
    seq_L = 1
    input_size = 28 * 28

    # fake data
    x = jax.random.randint(rng, (batch_size, input_size), 1, 100)
    # print(f"x:\n{x}\n")
    print(f"x shape:\n{x.shape}\n")
    x = jnp.expand_dims(x, axis=-1)
    vals = jnp.ones((batch_size, input_size, batch_size - 1)) * input_size
    # print(f"vals:\n{vals}\n")
    print(f"vals shape:\n{vals.shape}\n")
    x = jnp.concatenate([x, vals], axis=-1)

    # model
    model = SimpleRNN(
        input_size=(28 * 28),
        hidden_size=hidden_size,
        num_layers=3,
        output_size=10,
        bias=True,
        activation="tanh",
    )

    # get model params
    params = model.init(
        key,
        model.initialize_carry(
            rng=subkey,
            batch_dims=(batch_size,),
            size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        input=x,
    )

    out = model.apply(
        params,
        model.initialize_carry(
            rng=subsubkey,
            batch_dims=(batch_size,),
            size=hidden_size,
            init_fn=nn.initializers.zeros,
        ),
        x,
    )

    xshape = out.shape
    return x, xshape


def tester():
    for i in range(1, 100):
        testx, xdims = test()
        if i % 10 == 0:
            print(f"output array:\n{testx[i]}\n")
            print(f"output array shape:\n{xdims}\n")
        assert xdims == (64, 10)
    print("Size test: passed.")


def main():
    tester()


if __name__ == "__main__":
    main()
