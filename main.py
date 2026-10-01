from model import BabyChilla
import flax.nnx as nnx


def main():
    batch = 5
    seq_len = 12
    emb_dim=4
    num_heads=2
    ffw_dim=8
    output_dim = emb_dim
    rngs=nnx.Rngs(0)

    print("Hello from chinchilla-scaling!")


if __name__ == "__main__":
    main()
