from flax import nnx

from model import BabyChillaModel
from train import BabyChilla



def main():
    batch_size = 64
    seq_len = 12
    num_epochs = 5
    rngs=nnx.Rngs(0)

    # 11,842,400 model params 
    vocab_size = 14 
    num_heads = 16
    num_layers = 8
    emb_dim=400
    ffw_dim=4*emb_dim
    output_size = 1999
    
    model_instance = BabyChillaModel(emb_dim, num_heads, ffw_dim, num_layers, rngs=rngs, output_size=output_size)
    # print(f"# Params excluding embedding: {model_instance.get_num_params()}")
    # print(f"# Params with embedding: {model_instance.get_num_params(exclude_embedding=False)}")
    model = BabyChilla(model=model_instance, batch_size=batch_size, num_epochs=num_epochs)
    model.train()


if __name__ == "__main__":
    main()
