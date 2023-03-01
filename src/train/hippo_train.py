from typing import Callable, Tuple

import jax
import optax
import wandb
from functools import partial
from flax.training import train_state
from jax import numpy as jnp

from src.train.task import Task
from src.train.trainer import Trainer


class HiPPOTrainer(Trainer):
    """
    Class to manage model training and feature extraction.
    """

    def __init__(self, task: Task):
        self.task = task
        key = jax.random.PRNGKey(self.task.seed)
        self.keys = jax.random.split(key, num=5)

    def init_train_state(self, params) -> train_state.TrainState:
        # Get the model
        model = self.task.model

        # Get the optimizer
        optimizer = self.task.optimizer.tx

        # Create a State
        return train_state.TrainState.create(
            apply_fn=model.apply, params=params, tx=optimizer
        )

    @partial(jax.jit, static_argnums=(0,))
    def update_model(self, state, grads):
        return state.apply_gradients(grads=grads)

    def train(self, state):

        train_batch_metrics = []
        for batch_idx, (train_data, train_labels) in enumerate(
            self.task.dataset.train_loader
        ):
            train_data = jnp.array(train_data.numpy())
            train_labels = jnp.array(train_labels.numpy())
            grads, state, metrics = self.step(state, (train_data, train_labels))
            state = self.update_model(state, grads)
            train_batch_metrics.append(metrics)

        eval_batch_metrics = []
        for test_data, test_labels in self.task.dataset.train_loader:
            test_data = jnp.array(test_data.numpy())
            test_labels = jnp.array(test_labels.numpy())
            metrics = self.eval(state, (test_data, test_labels))
            eval_batch_metrics.append(metrics)

        return state, train_batch_metrics, eval_batch_metrics

    def run(self, epochs: int, batch_size: int):
        params = self.task.model.init(
            self.keys[0], f=jnp.zeros(shape=(batch_size, 784, 1))
        )
        state = self.init_train_state(params)
        for epoch in range(epochs):
            state, train_batch_metrics, eval_batch_metrics = self.train(state)
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

    @partial(jax.jit, static_argnums=(0,))
    def step(
        self,
        state: train_state.TrainState,
        batch: Tuple[jnp.ndarray, jnp.ndarray],
    ):
        data, label = batch

        def loss_fn(params):

            logits = state.apply_fn({"params": params}, f=data, init_state=None)

            if len(self.task.pipeline.pipeline) > 0:
                for fn in self.task.pipeline.pipeline:
                    label = fn(label)

            loss = self.task.loss.apply(logits, label)

            return loss, logits

        gradient_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, logits), grads = gradient_fn(state.params)
        metrics = self.compute_metrics(logits=logits, labels=label)

        return grads, state, metrics

    @partial(jax.jit, static_argnums=(0,))
    def eval(self, state, batch):
        data, label = batch
        logits = state.apply_fn({"params": state.params}, data)
        metrics = self.compute_metrics(logits=logits, labels=label)
        return metrics

    def compute_metrics(self, logits, labels):
        loss = self.task.loss.apply(logits, labels)
        accuracy = jnp.mean(jnp.argmax(logits, -1) == labels)
        metrics = {"loss": loss, "accuracy": accuracy}
        return metrics
