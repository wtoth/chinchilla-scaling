from flax import nnx

from model import BabyChillaModel
from train import BabyChilla


def main():
    batch = 64
    seq_len = 12
    emb_dim=128
    num_heads=8
    ffw_dim=128
    output_dim = emb_dim
    num_layers = 2
    num_epochs = 5
    rngs=nnx.Rngs(0)

    model_instance = BabyChillaModel(emb_dim, num_heads, ffw_dim, num_layers, rngs=rngs)
    model = BabyChilla(model=model_instance, batch_size=batch, num_epochs=num_epochs)
    model.train()


if __name__ == "__main__":
    main()
