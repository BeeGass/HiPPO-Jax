from typing import Callable

import jax
import optax
import wandb
from flax.training import train_state
from jax import numpy as jnp

from src.train.task import Task
from src.train.trainer import Trainer


class HiPPOTrainer(Trainer):
    """
    Class to manage model training and feature extraction.
    """

    task: Task

    def init_train_state(self, params) -> train_state.TrainState:
        # Get the model
        model = self.task.model

        # Get the optimizer
        optimizer = self.task.optimizer.tx

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
        for batch_idx, (train_data, train_labels) in enumerate(
            self.task.dataset.train_loader
        ):
            grads, state, metrics = self.step(state, (train_data, train_labels))
            state = self.update_model(state, grads)
            train_batch_metrics.append(metrics)

        eval_batch_metrics = []
        for test_data, test_labels in self.task.dataset.train_loader:
            metrics = self.eval(state, (test_data, test_labels))
            eval_batch_metrics.append(metrics)

        return state, train_batch_metrics, eval_batch_metrics

    def epoch(self, epochs: int, batch_size: int, data_gen):
        for epoch in range(epochs):
            state, train_batch_metrics, eval_batch_metrics = self.train(
                batch_size, data_gen
            )
            wandb.log(
                {
                    "Train Loss": train_batch_metrics["loss"],
                    "Train Accuracy": train_batch_metrics["accuracy"],
                    "Validation Loss": eval_batch_metrics["loss"],
                    "Validation Accuracy": eval_batch_metrics["accuracy"],
                },
                step=epoch,
            )

        return state

    @jax.jit
    def step(
        self,
        state: train_state.TrainState,
        batch: jnp.ndarray,
    ):
        data, label = batch

        def loss_fn(params):

            logits = state.apply_fn({"params": params}, f=data, init_state=None)

            if self.task.pipeline.pipeline != []:
                for fn in self.task.pipeline.pipeline:
                    logits = fn(logits)

            loss = self.task.loss.apply(logits, label)

            return loss, logits

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
