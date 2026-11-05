import os
import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

import flax.nnx as nnx
import optax
from orbax.checkpoint import v1 as ocp
from etils import epath

import wandb
from tqdm import tqdm
from dataset import ChinchillaDataset
from torch.utils.data import Dataset, DataLoader

class BabyChilla:
    def __init__(self, model, batch_size, num_epochs, log=True, log_every=100):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.log = log
        self.log_every = log_every
        self.learning_rate = 1e-3
        self.training_checkpointer = ocp.training
        self.weight_dir = os.path.abspath("weight_dir")
        self.optimizer = nnx.Optimizer(model, optax.adam(self.learning_rate), wrt=nnx.Param)
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )
        self.val_metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )


    def train(self, dataset="data/train.csv"):
        if self.log:
            self.init_logging(self.batch_size, self.learning_rate, self.num_epochs)

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
                    loss = self.train_step(
                        # self.model,
                        # self.optimizer,
                        # self.metrics,
                        jax.device_put(
                            (inputs.T, labels), # P("batch", None)
                        ),
                    )
                    global_step = epoch * len(train_dataloader) + i
                    if self.log and global_step % self.log_every == 0:
                        wandb.log({"train/loss": float(loss), "epoch": epoch}, step=global_step)

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
                if self.log:
                    wandb.log({"val/loss": float(metrics_history["val_loss"][-1])}, step=(epoch + 1) * len(train_dataloader))

                # Save the combined state
                state = {
                    'params': nnx.state(self.model),
                    'optimizer': nnx.state(self.optimizer, nnx.optimizer.OptState),
                }
                # ckptr.save(step, state)
                step += 1
        if self.log:
            wandb.finish()


    def loss_fn(self, model, batch):
        logits, total_aux_loss = model(batch[0])
        loss = optax.softmax_cross_entropy_with_integer_labels(logits=logits, labels=batch[1]).mean()
        loss += total_aux_loss # add aux loss
        # print(loss)
        return loss, logits

    # Define the training step with the `flax.nnx.jit` transformation decorator.
    # @nnx.jit(static_argnums=0)
    # @nnx.jit(donate_argnums=(1,))
    def train_step(self, batch):
        @nnx.jit(static_argnames="loss_fn")
        def _jitted_step(model, loss_fn, metrics, optimizer, batch):
            grad_fn = nnx.value_and_grad(loss_fn, has_aux=True)
            (loss, logits), grads = grad_fn(model, batch)
            metrics.update(loss=loss, logits=logits, labels=batch[1])
            optimizer.update(model, grads)
            return loss

        return _jitted_step(self.model, self.loss_fn, self.metrics, self.optimizer, batch)

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

    def init_logging(self, batch_size, learning_rate, num_epochs):
        wandb_log = wandb.init(
            entity="wtoth21",
            # Set the wandb project where this run will be logged.
            project="ChinchillaScaling",
            # Track hyperparameters and run metadata.
            config={
                "architecture": "Transformer",
                "MoE": self.model.MoE,
                "dataset": "Chinchilla",
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "epochs": num_epochs,
            })
        return wandb_log