from typing import Callable
import optax
import jax
import wandb
from jax import numpy as jnp
from flax.training import train_state

from src.train.task import Task


class Trainer:
    """
    Class to manage model training and feature extraction.
    """

    def init_train_state(self, params) -> train_state.TrainState:
        raise NotImplementedError()

    def train(self):
        raise NotImplementedError()

    def epoch(self):
        raise NotImplementedError()

    def step(self):
        raise NotImplementedError()

    def eval(self):
        raise NotImplementedError()


class HiPPOTrainer(Trainer):
    """
    Class to manage model training and feature extraction.
    """

    task: Task

    def init_train_state(self, params) -> train_state.TrainState:
        # Get the model
        model = self.task.model

        # Get the optimizer
        optimizer = self.task.optimizer

        # Create a State
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )

    @jax.jit
    def update_model(self, state, grads):
        return state.apply_gradients(grads=grads)

    def train(self, batch_size: int, data_gen):
        state = None

        train_batch_metrics = []
        for idx in range(batch_size):
            batch = next(data_gen)
            grads, state, metrics = self.step(state, batch)
            state = self.update_model(state, grads)
            train_batch_metrics.append(metrics)

        eval_batch_metrics = []
        for idx in range(batch_size):
            batch = next(data_gen)
            metrics = self.eval(state, batch)
            eval_batch_metrics.append(metrics)

        return state, train_batch_metrics, eval_batch_metrics

    def epoch(self, epochs: int, batch_size: int, data_gen):
        for epoch in range(epochs):
            state, train_batch_metrics, eval_batch_metrics = self.train(
                batch_size, data_gen
            )
            self.log_metrics(epoch, train_batch_metrics, eval_batch_metrics)

        return state

    @jax.jit
    def step(
        self, state: train_state.TrainState, batch: jnp.ndarray, loss_fn: Callable
    ):
        data, label = batch

        loss, logits = loss_fn(state, params, data, label)

        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = gradient_fn(state.params)
        state = state.apply_gradients(grads=grads)
        metrics = compute_metrics(logits=logits, labels=label)
        return grads, state, metrics

    @jax.jit
    def eval(self, state, batch):
        data, label = batch
        logits = state.apply_fn({"params": state.params}, data)
        metrics = compute_metrics(logits=logits, labels=label)
        return metrics

    def log_metrics(self):
        # Log Metrics to Weights & Biases
        wandb.log(
            {
                "Train Loss": train_batch_metrics["loss"],
                "Train Accuracy": train_batch_metrics["accuracy"],
                "Validation Loss": eval_batch_metrics["loss"],
                "Validation Accuracy": eval_batch_metrics["accuracy"],
            },
            step=epoch,
        )
