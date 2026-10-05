import os
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

import flax.nnx as nnx
import optax
from orbax.checkpoint import v1 as ocp
from etils import epath

from tqdm import tqdm
from dataset import ChinchillaDataset
from torch.utils.data import Dataset, DataLoader

class BabyChilla:
    def __init__(self, model, batch_size, num_epochs):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.training_checkpointer = ocp.training
        self.weight_dir = os.path.abspath("weight_dir")
        self.optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )
        self.val_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

    def train(self, dataset="data/train.csv"):

        with self.training_checkpointer.Checkpointer(self.weight_dir,) as ckptr:
            training_dataset = ChinchillaDataset(problems_file=dataset)
            train_dataloader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)

            val_dataset = ChinchillaDataset(problems_file="data/val.csv")
            val_dataloader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True)

            metrics_history = {
                "train_loss": [],
                "val_loss": [],
            }

            # ??????????
            prep_target_batch = jax.vmap(
                lambda tokens: jnp.concatenate((tokens[1:], jnp.array([0])))
            )

            step = 0
            for epoch in range(self.num_epochs):
                # start_time = time.time()
                for i, (inputs, labels) in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                    # print(inputs, labels)
                    inputs, labels = jnp.array(inputs, dtype=int), jnp.array(labels, dtype=int) # convert to jax.numpy arrays
                # for batch in train_dataloader:
                    if len(inputs) % len(jax.devices()) != 0: 
                        continue  # skip the remaining elements
                    # input_batch = jnp.stack(batch).T # ???????S
                    # target_batch = prep_target_batch(input_batch) # ???????
                    self.train_step(
                        # self.model,
                        # self.optimizer,
                        # self.metrics,
                        jax.device_put(
                            (inputs.T, labels), # P("batch", None)
                        ),
                    )

                for metric, value in self.metrics.compute().items():
                    metrics_history[f"train_{metric}"].append(value)
                self.metrics.reset()

                # elapsed_time = time.time() - start_time
                print(
                    # f"\n\nStep {step + 1}, Loss: {metrics_history['train_loss'][-1]}" # , Elapsed Time: {elapsed_time:.2f} seconds"
                    f"Training Loss: {metrics_history['train_loss'][-1]}" # , Elapsed Time: {elapsed_time:.2f} seconds" 
                )
                # start_time = time.time()

                self.evaluate_model(val_dataloader)
                for metric, value in self.val_metrics.compute().items():
                    metrics_history[f"val_{metric}"].append(value)
                    self.val_metrics.reset()
                print(
                    # f"\n\nVal Step {step + 1}, Val Loss: {metrics_history['val_loss'][-1]}" # , Elapsed Time: {elapsed_time:.2f} seconds"
                    f"Validation Loss: {metrics_history['val_loss'][-1]}" # , Elapsed Time: {elapsed_time:.2f} seconds"
                )

                # Save the combined state
                state = {
                    'params': nnx.state(self.model),
                    'optimizer': nnx.state(self.optimizer, nnx.optimizer.OptState),
                }
                # ckptr.save(step, state)
                step += 1



    def loss_fn(self, model, batch):
        logits = model(batch[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
        # print(loss)
        return loss, logits

    # Define the training step with the `flax.nnx.jit` transformation decorator.
    # @nnx.jit()
    def train_step(self, batch):
        grad_fn = nnx.value_and_grad(self.loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(self.model, batch)
        self.metrics.update(loss=loss, logits=logits, lables=batch[1])
        self.optimizer.update(self.model, grads)

    def evaluate_model(self, val_dataloader):
        # Computes the metrics on the training and test sets after each training epoch.
        self.val_metrics.reset()  # Reset the eval metrics
        for (inputs, labels) in val_dataloader:
            inputs, labels = jnp.array(inputs, dtype=int), jnp.array(labels, dtype=int) # convert to jax.numpy arrays
            self.eval_step(jax.device_put(
                                    (inputs.T, labels), # P("batch", None)
                                ),)

    def eval_step(self, batch: tuple[jax.Array, jax.Array]):
        (loss, logits) = self.loss_fn(self.model, batch)
        self.val_metrics.update(
            loss=loss,
            logits=logits,
            labels=batch[1],
        )