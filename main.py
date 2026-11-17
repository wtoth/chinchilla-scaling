import argparse
import os

from flax import nnx

from model import BabyChillaModel
from train import BabyChilla


def parse_args():
    parser = argparse.ArgumentParser(description="Train BabyChilla")

    # Data
    parser.add_argument("--dataset-dir", type=str, default="data",
                        help="Directory containing train.csv and val.csv")

    # Training
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=12)
    parser.add_argument("--num-epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-log", action="store_true", help="Disable wandb logging")

    # Model (defaults = 11,842,400 params)
    parser.add_argument("--vocab-size", type=int, default=14)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=8)
    parser.add_argument("--emb-dim", type=int, default=400)
    parser.add_argument("--ffw-dim", type=int, default=None, help="Defaults to 4 * emb-dim")
    parser.add_argument("--output-size", type=int, default=1999)
    parser.add_argument("--no-moe", action="store_true", help="Disable MoE (default: MoE on)")

    return parser.parse_args()


def main():
    args = parse_args()

    batch_size = args.batch_size
    seq_len = args.seq_len
    num_epochs = args.num_epochs
    rngs = nnx.Rngs(args.seed)

    vocab_size = args.vocab_size
    num_heads = args.num_heads
    num_layers = args.num_layers
    emb_dim = args.emb_dim
    ffw_dim = args.ffw_dim if args.ffw_dim is not None else 4 * emb_dim
    output_size = args.output_size

    train_path = os.path.join(args.dataset_dir, "train.csv")
    val_path = os.path.join(args.dataset_dir, "val.csv")

    model_instance = BabyChillaModel(emb_dim, num_heads, ffw_dim, num_layers, MoE=not args.no_moe, rngs=rngs, output_size=output_size)
    # print(f"# Params excluding embedding: {model_instance.get_num_params()}")
    # print(f"# Params with embedding: {model_instance.get_num_params(exclude_embedding=False)}")
    model = BabyChilla(model=model_instance, batch_size=batch_size, num_epochs=num_epochs, log=not args.no_log)
    model.train(dataset=train_path, val_dataset=val_path)


if __name__ == "__main__":
    main()