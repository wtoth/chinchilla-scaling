import jax
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, reshard

import flax.nnx as nnx
import optax

from tqdm import tqdm
from dataset import ChinchillaDataset
from torch.utils.data import Dataset, DataLoader

class BabyChilla:
    def __init__(self, model, batch_size, num_epochs):
        self.model = model
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
        self.metrics = nnx.MultiMetric(
            loss=nnx.metrics.Average("loss"),
        )

    def train(self, dataset="data/train.csv"):
        training_dataset = ChinchillaDataset(problems_file=dataset)
        train_dataloader = DataLoader(training_dataset, batch_size=self.batch_size, shuffle=True)

        metrics_history = {
            "train_loss": [],
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

                if (step + 1) % 200 == 0:
                    for metric, value in self.metrics.compute().items():
                        metrics_history[f"train_{metric}"].append(value)
                    self.metrics.reset()

                    # elapsed_time = time.time() - start_time
                    print(
                        f"\n\nStep {step + 1}, Loss: {metrics_history['train_loss'][-1]}" # , Elapsed Time: {elapsed_time:.2f} seconds"
                    )
                    # start_time = time.time()

                step += 1


    def loss_fn(self, model, batch):
        logits = model(batch[0])
        print(nnx.softmax(logits, axis=-1))
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