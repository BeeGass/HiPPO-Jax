import jax
from jax import jit, vmap
import jax.numpy as jnp

import numpy as np
import optax

import flax
from flax import linen as nn
from flax.training import train_state
from flax.linen.activation import tanh
from flax.linen.activation import sigmoid

# from flax.linen.recurrent import RNNCellBase
from src.models.rnn.cells import GRUCell, HiPPOCell, LSTMCell, RNNCell
from src.models.rnn.rnn import (
    OneToManyRNN,
    ManyToOneRNN,
    ManyToManyRNN,
    DeepRNN,
    BidirectionalRNN,
)
from src.models.hippo.hippo import HiPPO
from src.models.hippo.transition import TransMatrix
from src.data.process import moving_window, rolling_window

import torch
from torchvision import datasets, transforms

import time
from typing import Any, Callable, Sequence, Optional, Tuple, Union
from collections import defaultdict
from omegaconf import DictConfig, OmegaConf
import wandb
import hydra


def get_datasets(cfg):
    # download and transform train dataset
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../datasets/mnist_data",
            download=True,
            train=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                ]
            ),
        ),
        batch_size=cfg.training.params.batch_size,
        shuffle=True,
    )

    # download and transform test dataset
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            "../../datasets/mnist_data",
            download=True,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),  # first, convert image to PyTorch tensor
                ]
            ),
        ),
        batch_size=cfg.training.params.batch_size,
        shuffle=True,
    )
    return train_loader, test_loader


def pick_rnn_cell(cfg, stack_number):
    # set rnn cell from rnn_type
    rnn_list = []
    if cfg.models.recurrent.cell_type == "rnn":
        rnn_list = [
            RNNCell(
                input_size=cfg.models.recurrent.cell.rnn.input_size,
                hidden_size=cfg.models.recurrent.cell.rnn.hidden_size,
                bias=cfg.models.recurrent.cell.rnn.bias,
                param_dtype=jnp.float32,
                activation_fn=tanh,
            )
            for _ in range(stack_number)
        ]

    elif cfg.models.recurrent.cell_type == "lstm":
        rnn_list = [
            LSTMCell(
                input_size=cfg.models.recurrent.cell.lstm.input_size,
                hidden_size=cfg.models.recurrent.cell.lstm.hidden_size,
                bias=cfg.models.recurrent.cell.lstm.bias,
                param_dtype=jnp.float32,
                gate_fn=sigmoid,
                activation_fn=tanh,
            )
            for _ in range(stack_number)
        ]

    elif cfg.models.recurrent.cell_type == "gru":
        rnn_list = [
            GRUCell(
                input_size=cfg.models.recurrent.cell.gru.input_size,
                hidden_size=cfg.models.recurrent.cell.gru.hidden_size,
                bias=cfg.models.recurrent.cell.gru.bias,
                param_dtype=jnp.float32,
                gate_fn=sigmoid,
                activation_fn=tanh,
            )
            for _ in range(stack_number)
        ]

    elif cfg.models.recurrent.cell_type == "hippo":
        rnn_list = [
            HiPPOCell(
                input_size=cfg.models.recurrent.cell.hippo.input_size,
                hidden_size=cfg.models.recurrent.cell.hippo.hidden_size,
                bias=cfg.models.recurrent.cell.hippo.bias,
                param_dtype=jnp.float32,
                gate_fn=sigmoid,
                activation_fn=tanh,
                measure=cfg.models.recurrent.cell.hippo.measure,
                lambda_n=cfg.models.recurrent.cell.hippo.lambda_n,
                fourier_type=cfg.models.recurrent.cell.hippo.fourier_type,
                alpha=cfg.models.recurrent.cell.hippo.alpha,
                beta=cfg.models.recurrent.cell.hippo.beta,
                rnn_cell=GRUCell,
            )
            for _ in range(stack_number)
        ]

    else:
        raise ValueError("Unknown rnn type")

    return rnn_list


def pick_model(key, cfg):
    # set model from net_type
    the_key, subkey = jax.random.split(key)
    model = None
    params = None
    init_carry = None

    if cfg.models.recurrent.architecture == "deep rnn":
        stack_number = cfg.models.recurrent.architectures.deep_rnn.stack_number
        rnn_list = pick_rnn_cell(cfg, stack_number)
        model = DeepRNN(
            output_size=cfg.models.recurrent.architectures.deep_rnn.output_size,
            layers=rnn_list,
            skip_connections=cfg.models.recurrent.architectures.deep_rnn.skip_connections,
        )
        init_carry = model.initialize_carry(
            rng=the_key,
            batch_size=(cfg.training.params.batch_size,),
            hidden_size=cfg.models.recurrent.architectures.deep_rnn.hidden_size,
            init_fn=nn.initializers.zeros,
        )
        x = jnp.zeros((cfg.training.params.batch_size, cfg.training.input_size))
        input = vmap(moving_window, in_axes=(0, None))(x, cfg.training.input_length)
        params = model.init(subkey, carry=init_carry, input=input)["params"]

    elif cfg.models.recurrent.architecture == "bidirectional rnn":
        stack_number = cfg.models.recurrent.architectures.bidirectional.stack_number
        rnn_list = pick_rnn_cell(cfg, stack_number)
        raise NotImplementedError("Bidirectional RNN not implemented yet")

    elif cfg.models.recurrent.architecture == "one to many rnn":
        stack_number = cfg.models.recurrent.architectures.onetomany.stack_number
        rnn_list = pick_rnn_cell(cfg, stack_number)
        model = OneToManyRNN(
            output_size=cfg.models.recurrent.architectures.onetomany.output_size,
            layer=rnn_list,
        )
        init_carry = model.initialize_carry(
            rng=the_key,
            batch_size=(cfg.training.params.batch_size,),
            hidden_size=cfg.models.recurrent.architectures.onetomany.hidden_size,
            init_fn=nn.initializers.zeros,
        )
        x = jnp.zeros((cfg.training.params.batch_size, cfg.training.input_size))
        input = vmap(moving_window, in_axes=(0, None))(x, cfg.training.input_length)
        params = model.init(
            subkey,
            carry=init_carry,
            input=input,
            teacher_forcing=cfg.models.recurrent.architectures.train_tf,
        )["params"]

    elif cfg.models.recurrent.architecture == "many to one rnn":
        stack_number = cfg.models.recurrent.architectures.manytoone.stack_number
        rnn_list = pick_rnn_cell(cfg, stack_number)
        model = ManyToOneRNN(
            output_size=cfg.models.recurrent.architectures.manytoone.output_size,
            layer=rnn_list,
        )
        init_carry = model.initialize_carry(
            rng=the_key,
            batch_size=(cfg.training.params.batch_size,),
            hidden_size=cfg.models.recurrent.architectures.manytoone.hidden_size,
            init_fn=nn.initializers.zeros,
        )
        x = jnp.zeros((cfg.training.params.batch_size, cfg.training.input_size))
        input = vmap(moving_window, in_axes=(0, None))(x, cfg.training.input_length)
        params = model.init(subkey, carry=init_carry, input=input)["params"]

    elif cfg.models.recurrent.architecture == "many to many rnn":
        stack_number = cfg.models.recurrent.architectures.manytomany.stack_number
        rnn_list = pick_rnn_cell(cfg, stack_number)
        model = ManyToManyRNN(
            output_size=cfg.models.recurrent.architectures.manytomany.output_size,
            layer=rnn_list,
        )
        init_carry = model.initialize_carry(
            rng=the_key,
            batch_size=(cfg.training.params.batch_size,),
            hidden_size=cfg.models.recurrent.architectures.manytomany.hidden_size,
            init_fn=nn.initializers.zeros,
        )
        x = jnp.zeros((cfg.training.params.batch_size, cfg.training.input_size))
        input = vmap(moving_window, in_axes=(0, None))(x, cfg.training.input_length)
        params = model.init(subkey, carry=init_carry, input=input)["params"]

    elif cfg.models.recurrent.architecture == "hippo":
        L = cfg.training.input_length
        hippo_matrices = TransMatrix(
            N=cfg.models.state_spaces.transition_matrix.n,
            measure=cfg.models.state_spaces.transition_matrix.measure,
            lambda_n=cfg.models.state_spaces.transition_matrix.lambda_n,
            fourier_type=cfg.models.state_spaces.transition_matrix.fourier_type,
            alpha=cfg.models.state_spaces.transition_matrix.alpha,
            beta=cfg.models.state_spaces.transition_matrix.beta,
        )
        model = HiPPO(
            N=cfg.models.state_spaces.HiPPO.n,
            max_length=L,
            step=1.0 / L,
            GBT_alpha=cfg.models.state_spaces.HiPPO.GBT_alpha,
            seq_L=L,
            A=hippo_matrices.A_matrix,
            B=hippo_matrices.B_matrix,
            measure=cfg.models.state_spaces.HiPPO.measure,
        )
        params = model.init(f, init_state=None, t_step=0, kernel=False)

    elif cfg.models.recurrent.architecture == "s4":
        raise NotImplementedError
        # model = S4()
        # params = model.init()

    else:
        raise ValueError("Unknown model type")

    return model, params, init_carry
    # return model, params


def preprocess_data(cfg, data):
    # preprocess data
    x = None
    if cfg.data.dataset.preprocess_data == "flatten":
        x = data.cpu().detach().numpy()
        x = jnp.asarray(data, dtype=jnp.float32)
        x = jnp.squeeze(x, axis=1)
        x = vmap(jnp.ravel, in_axes=0)(x)
        x = vmap(moving_window, in_axes=(0, None))(x, cfg["training"]["input_length"])

    return x


def preprocess_labels(cfg, labels):
    # preprocess data
    y = None
    if cfg.data.dataset.preprocess_labels == "one hot":
        y = labels.cpu().detach().numpy()
        y = jnp.asarray(y, dtype=jnp.float32)
        # y = jax.nn.one_hot(y, 10, dtype=jnp.float32)

    return y


def pick_optim(cfg, model, params):

    tx = None
    if cfg.training.params.optim == "adam":
        tx = optax.adamw(
            learning_rate=cfg.training.params.lr,
            weight_decay=cfg.training.params.weight_decay,
        )
    elif cfg.training.params.optim == "sgd":
        tx = optax.sgd(learning_rate=cfg.training.params.lr)
    else:
        raise ValueError("Unknown optimizer")

    # tx_state = tx.init(params)
    # print(f"tx_state: {tx_state}")

    return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    # , opt_state=tx_state


@jax.jit
def apply_model(state, carry, data, labels):
    """Computes gradients, loss and accuracy for a single batch."""

    def loss_fn(params):
        # jax.debug.print("params:\n{params}", params=params)

        logits = state.apply_fn({"params": params}, carry=carry, input=data)

        one_hot = jax.nn.one_hot(labels, 10, dtype=jnp.float32)
        loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=one_hot))

        # jax.debug.print("logits:\n{logits}", logits=logits)
        # jax.debug.print("labels:\n{labels}", labels=labels)
        # jax.debug.print("one_hot:\n{one_hot}", one_hot=one_hot)
        # jax.debug.print("loss:\n{loss}", loss=loss)

        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, logits), grads = grad_fn(state.params)
    accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)

    return grads, loss, accuracy


@jax.jit
def update_model(state, grads):
    return state.apply_gradients(grads=grads)


@hydra.main(config_path="../../config", config_name="config")
def recurrent_train(
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
        project="BeeGass-HiPPOs", entity="beegass", config=cfg
    ):  # initialize wandb project for logging

        # get keys for parameters
        seed = cfg.training.seed
        key = jax.random.PRNGKey(seed)

        num_copies = cfg.training.key_num
        a_key, subkey = jax.random.split(key, num=num_copies)

        # get train and test datasets
        train_loader, test_loader = get_datasets(cfg)
        print(f"got dataset")

        # pick a model
        model, params, carry = pick_model(a_key, cfg)
        print(f"got model and params")

        # pick an optimizer
        state = pick_optim(cfg, model, params)
        print(f"got optimizer state")

        # pick a scheduler
        # TODO: implement choice of scheduler

        # pick a loss function
        # TODO: implement choice of loss function

        epoch_loss = []
        epoch_accuracy = []

        print(f"starting training loop")
        # Loop over the training epochs
        for epoch in range(cfg.training.params.num_epochs):
            start_time = time.time()
            for batch_id, (train_data, train_labels) in enumerate(train_loader):
                data = preprocess_data(cfg, train_data)
                labels = preprocess_labels(cfg, train_labels)
                # carry = model.initialize_carry(
                #     rng=subkey,
                #     batch_size=(cfg["training"]["batch_size"],),
                #     hidden_size=cfg["models"]["deep_rnn"]["hidden_size"],
                # )
                # grads, loss, accuracy = apply_model(
                #     state=state, carry=None, data=data, labels=labels
                # )
                grads, loss, accuracy = apply_model(
                    state=state,
                    carry=carry,
                    data=data,
                    labels=labels,
                )
                state = update_model(state, grads)
                epoch_loss.append(loss)
                epoch_accuracy.append(accuracy)

            # train loss and accuracy for current epoch
            train_loss = jnp.mean(jnp.array(epoch_loss))
            train_accuracy = jnp.mean(jnp.array(epoch_accuracy))
            wandb.log(
                {"train_loss": train_loss, "train_accuracy": train_accuracy}, step=epoch
            )

            epoch_test_loss = []
            epoch_test_accuracy = []

            for data, target in test_loader:
                data = preprocess_data(cfg, train_data)
                target = preprocess_labels(cfg, target)

                # test loss for current epoch
                # _, test_loss, test_accuracy = apply_model(
                #     state=state, carry=None, data=data, labels=target
                # )
                _, test_loss, test_accuracy = apply_model(
                    state=state,
                    carry=carry,
                    data=data,
                    labels=target,
                )
                epoch_test_loss.append(test_loss)
                epoch_test_accuracy.append(test_accuracy)

            test_epoch_loss = jnp.mean(jnp.array(epoch_test_loss))
            test_epoch_accuracy = jnp.mean(jnp.array(epoch_test_accuracy))
            wandb.log(
                {"test_loss": test_epoch_loss, "test_accuracy": test_epoch_accuracy},
                step=epoch,
            )

            epoch_time = time.time() - start_time
            print(f"Epoch {epoch + 1} in {epoch_time:.2f} sec")
            print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
            print(
                f"Test Loss: {test_epoch_loss:.4f}, Test Accuracy: {test_epoch_accuracy:.4f}"
            )

        return state
