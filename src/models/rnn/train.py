import jax
import jax.numpy as jnp
import numpy as np
import flax
import optax

from flax import linen as nn
from flax.training import train_state

# from flax.linen.recurrent import RNNCellBase
from src.models.rnn.cells import GRUCell, HiPPOCell, LSTMCell, RNNCell
from src.models.rnn.rnn import DeepRNN
from src.models.hippo.hippo import HiPPO
from src.models.hippo.transition import TransMatrix

import time
from typing import Any, Callable, Sequence, Optional, Tuple, Union
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import wandb
import hydra
import tensorflow_datasets as tfds


def get_datasets():
    """Load MNIST train and test datasets into memory."""
    ds_builder = tfds.builder("mnist")
    ds_builder.download_and_prepare()
    train_ds = tfds.as_numpy(ds_builder.as_dataset(split="train", batch_size=-1))
    test_ds = tfds.as_numpy(ds_builder.as_dataset(split="test", batch_size=-1))
    train_ds["image"] = jnp.float32(train_ds["image"]) / 255.0
    test_ds["image"] = jnp.float32(test_ds["image"]) / 255.0
    return train_ds, test_ds


def pick_rnn_cell(cfg):
    # set rnn cell from rnn_type
    rnn_list = []
    if cfg["models"]["cells"]["cell_type"] == "rnn":
        rnn_list = [
            RNNCell(
                input_size=cfg["models"]["cells"]["rnn"]["input_size"],
                hidden_size=cfg["models"]["cells"]["rnn"]["hidden_size"],
                bias=cfg["models"]["cells"]["rnn"]["bias"],
                param_dtype=cfg["models"]["cells"]["rnn"]["param_dtype"],
                activation_fn=cfg["models"]["cells"]["rnn"]["activation_fn"],
            )
            for _ in range(cfg["models"]["deep_rnn"]["stack_number"])
        ]

    elif cfg["models"]["cells"]["cell_type"] == "lstm":
        rnn_list = [
            LSTMCell(
                input_size=cfg["models"]["cells"]["gated_rnn"]["input_size"],
                hidden_size=cfg["models"]["cells"]["gated_rnn"]["hidden_size"],
                bias=cfg["models"]["cells"]["gated_rnn"]["bias"],
                param_dtype=cfg["models"]["cells"]["gated_rnn"]["param_dtype"],
                gate_fn=cfg["models"]["cells"]["gated_rnn"]["gate_fn"],
                activation_fn=cfg["models"]["cells"]["gated_rnn"]["activation_fn"],
            )
            for _ in range(cfg["models"]["deep_rnn"]["stack_number"])
        ]

    elif cfg["models"]["cells"]["cell_type"] == "gru":
        rnn_list = [
            GRUCell(
                input_size=cfg["models"]["cells"]["gated_rnn"]["input_size"],
                hidden_size=cfg["models"]["cells"]["gated_rnn"]["hidden_size"],
                bias=cfg["models"]["cells"]["gated_rnn"]["bias"],
                param_dtype=cfg["models"]["cells"]["gated_rnn"]["param_dtype"],
                gate_fn=cfg["models"]["cells"]["gated_rnn"]["gate_fn"],
                activation_fn=cfg["models"]["cells"]["gated_rnn"]["activation_fn"],
            )
            for _ in range(cfg["models"]["deep_rnn"]["stack_number"])
        ]

    elif cfg["models"]["cells"]["cell_type"] == "legs_lstm":
        rnn_list = [
            HiPPOCell(
                input_size=cfg["models"]["cells"]["hippo"]["input_size"],
                hidden_size=cfg["models"]["cells"]["hippo"]["hidden_size"],
                bias=cfg["models"]["cells"]["hippo"]["bias"],
                param_dtype=cfg["models"]["cells"]["hippo"]["param_dtype"],
                gate_fn=cfg["models"]["cells"]["hippo"]["gate_fn"],
                activation_fn=cfg["models"]["cells"]["hippo"]["activation_fn"],
                measure=cfg["models"]["cells"]["hippo"]["measure"],
                lambda_n=cfg["models"]["cells"]["hippo"]["lambda_n"],
                fourier_type=cfg["models"]["cells"]["hippo"]["fourier_type"],
                alpha=cfg["models"]["cells"]["hippo"]["alpha"],
                beta=cfg["models"]["cells"]["hippo"]["beta"],
                rnn_cell=cfg["models"]["cells"]["hippo"]["rnn_cell"],
            )
            for _ in range(cfg["models"]["deep_rnn"]["stack_number"])
        ]

    else:
        raise ValueError("Unknown rnn type")

    return rnn_list


def pick_model(key, cfg):
    # set model from net_type
    model = None
    params = None

    if cfg["models"]["model_type"] == "rnn":
        rnn_list = pick_rnn_cell(cfg)
        model = DeepRNN(
            output_size=cfg["models"]["deep_rnn"]["output_size"],
            layers=rnn_list,
            skip_connections=cfg["models"]["deep_rnn"]["skip_connections"],
        )
        init_carry = model.initialize_carry(
            rng=key,
            batch_size=(cfg["training"]["batch_size"],),
            hidden_size=cfg["models"]["deep_rnn"]["hidden_size"],
            init_fn=nn.initializers.zeros,
        )
        params = model.init(input, init_carry)

    elif cfg["models"]["model_type"] == "hippo":
        L = cfg["training"]["input_length"]
        hippo_matrices = TransMatrix(
            N=cfg["models"]["hippo"]["n"],
            measure=cfg["models"]["hippo"]["measure"],
            lambda_n=cfg["models"]["hippo"]["lambda_n"],
            fourier_type=cfg["models"]["hippo"]["fourier_type"],
            alpha=cfg["models"]["hippo"]["alpha"],
            beta=cfg["models"]["hippo"]["beta"],
        )
        model = HiPPO(
            N=cfg["models"]["hippo"]["n"],
            max_length=L,
            step=1.0 / L,
            GBT_alpha=cfg["models"]["hippo"]["GBT_alpha"],
            seq_L=L,
            A=hippo_matrices.A_matrix,
            B=hippo_matrices.B_matrix,
            measure=cfg["models"]["hippo"]["measure"],
        )
        params = model.init(f, init_state=None, t_step=0, kernel=False)

    elif cfg["models"]["model_type"] == "s4":
        raise NotImplementedError
        # model = S4()
        # params = model.init()

    else:
        raise ValueError("Unknown model type")

    return model, params


def preprocess_data(cfg, data):
    # preprocess data
    x = None
    if cfg["models"]["model_type"] == "rnn":
        batch_size = cfg["training"]["input_length"]
        seq_l = x.shape[-1]
        input_size = cfg["training"]["input_length"]
        array_shape = (batch_size, seq_l, input_size)
        x = jnp.expand_dims(data, -1)
        _x = jnp.ones(array_shape) * (input_size)
        x = jnp.concatenate([x, _x], axis=-1)

    elif cfg["models"]["model_type"] == "hippo":
        raise NotImplementedError

    elif cfg["models"]["model_type"] == "s4":
        raise NotImplementedError

    else:
        raise ValueError("Unknown model type to preprocess for")

    return x


def preprocess_labels(cfg, labels):
    # preprocess data
    x = None
    if cfg["models"]["model_type"] == "rnn":
        batch_size = cfg["training"]["input_length"]
        seq_l = x.shape[-1]
        input_size = cfg["training"]["input_length"]
        array_shape = (batch_size, seq_l, input_size)
        x = jnp.expand_dims(labels, -1)
        _x = jnp.ones(array_shape) * (input_size)
        x = jnp.concatenate([x, _x], axis=-1)

    elif cfg["models"]["model_type"] == "hippo":
        raise NotImplementedError

    elif cfg["models"]["model_type"] == "s4":
        raise NotImplementedError

    else:
        raise ValueError("Unknown model type to preprocess for")

    return x


def pick_optim(cfg, model, params):

    tx = None
    if cfg["training"]["optimizer"] == "adam":
        tx = optax.adam(
            learning_rate=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"]["weight_decay"],
        )
    elif cfg["training"]["optimizer"] == "sgd":
        tx = optax.sgd(learning_rate=cfg["training"]["learning_rate"])
    else:
        raise ValueError("Unknown optimizer")

    tx_state = tx.init(params)

    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx, opt_state=tx_state
    )


@jax.jit
def apply_model(state, data, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        logits = state.apply_fn({"params": params}, data)
        one_hot = jax.nn.one_hot(labels, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@hydra.main(config_path="config", config_name="train")
def _main(
    cfg: DictConfig,
) -> None:  # num_epochs, opt_state, net_type="RNN", train_key=None):
    """
    Implements a learning loop over epochs.

    Args:
        cfg: Hydra config

    Returns:
        None

    """
    with wandb.init(
        project="BeeGass-Sequential", entity="beegass", config=cfg
    ):  # initialize wandb project for logging

        # get keys for parameters
        seed = cfg["training"]["seed"]
        key = jax.random.PRNGKey(seed)

        num_copies = cfg["training"]["key_num"]
        keys = jax.random.split(key, num=num_copies)

        # get train and test datasets
        train_set, test_set = get_datasets()

        # pick a model
        model, params = pick_model(keys[1], cfg)

        # pick an optimizer
        state = pick_optim(cfg, model, params)

        # pick a scheduler
        # TODO: implement choice of scheduler

        # pick a loss function
        # TODO: implement choice of loss function

        # get dataset info for training loop (number of steps per epoch)
        train_set_size = len(train_set["image"])
        steps_per_epoch = train_set_size // cfg["training"]["batch_size"]

        perms = jax.random.permutation(keys[0], train_set_size)
        perms = perms[
            : steps_per_epoch * cfg["training"]["batch_size"]
        ]  # skip incomplete batch
        perms = perms.reshape((steps_per_epoch, cfg["training"]["batch_size"]))

        epoch_loss = []
        epoch_accuracy = []

        # Loop over the training epochs
        for epoch in range(cfg["training"]["num_epochs"]):
            # start_time = time.time()

            for perm in perms:
                train_data = train_set["image"][perm, ...]
                train_labels = train_set["label"][perm, ...]
                # x = preprocess_data(cfg, train_data)
                # y = preprocess_labels(cfg, train_labels)
                grads, loss, accuracy = apply_model(
                    state=state, data=train_data, labels=train_labels
                )
                state = update_model(state, grads)
                epoch_loss.append(loss)
                epoch_accuracy.append(accuracy)

            # epoch_time = time.time() - start_time

            # train loss for current epoch
            train_loss = np.mean(epoch_loss)
            train_accuracy = np.mean(epoch_accuracy)

            # test loss for current epoch
            _, test_loss, test_accuracy = apply_model(
                state=state, data=test_set["image"], labels=test_set["label"]
            )

            # TODO: add logging of metrics

        return state


def main():
    _main()


if __name__ == "__main__":
    main()
