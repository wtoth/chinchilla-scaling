import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

# ffw should equal embed_dim for simplicity

class BabyChilla(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, layers: int, rngs: nnx.Rngs, 
                 MoE: bool= False, max_len: int=12, vocab_size: int=12, output_size: int=1998, 
                 num_experts: int=12, active_experts: int=4):
        self.token_pos_emb = TokenAndPositionalEmbeddings(embed_dim, rngs=rngs, max_len=max_len, vocab_size=vocab_size)
        self.transformer_blocks = nnx.Sequential(*[TransformerBlock(embed_dim=embed_dim,
                                                                    num_heads=num_heads,
                                                                    ffw_dim=ffw_dim, 
                                                                    MoE=MoE,
                                                                    num_experts=num_experts,
                                                                    active_experts=active_experts,
                                                                    rngs=rngs) 
                                                                    for _ in range(layers)])
        self.output_layer = nnx.Linear(in_features=embed_dim, out_features=output_size, rngs=rngs)

    def __call__(self, x):
        x = self.token_pos_emb(x)
        x = self.transformer_blocks(x)
        # retrieve only one of the output points since we are doing classification instead of next token prediction
        # so the output is [num_batches, num_classes]
        x = x[:, 0] # idea from https://flax.readthedocs.io/en/stable/examples/vit_training.html
        out = self.output_layer(x)
        return out

# Transformer blocks made up for Transformer layers followed by MLP or MoE
class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, MoE: bool, num_experts: int, active_experts: int, rngs: nnx.Rngs):
        self.transformer = TransformerLayer(embed_dim, num_heads, rngs=rngs)
        
        if not MoE:
            self.ffw = MLP(input_dim=embed_dim, ffw_dim=ffw_dim, output_dim=embed_dim, rngs=rngs)
        else:
            self.ffw = MoE(ffw_dim, num_experts, active_experts)

    def __call__(self, inputs):
        transformer_output = self.transformer(inputs) + inputs # residual connection
        ffw_output = self.ffw(transformer_output) + transformer_output # ffw residual connection? 
        return ffw_output

class TransformerLayer(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, rngs: nnx.Rngs,):
        assert embed_dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.layer_norm = nnx.LayerNorm(epsilon=1e-6, num_features=embed_dim, rngs=rngs) # BTD

        self.q_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs, use_bias=False) # BTNH
        self.k_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs, use_bias=False) # BSKH
        self.v_proj = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs, use_bias=False) # BSKH

        self.output = nnx.Linear(in_features=embed_dim, out_features=embed_dim, rngs=rngs)

    def __call__(self, inputs):
        b, t, d = inputs.shape
        input_norm = self.layer_norm(inputs)
        q = self.q_proj(input_norm)
        k = self.q_proj(input_norm)
        v = self.q_proj(input_norm)

        q = q.reshape(b, t, self.num_heads, self.head_dim) # BTD -> BTNH
        k = k.reshape(b, t, self.num_heads, self.head_dim) # BSD -> BSNH
        v = v.reshape(b, t, self.num_heads, self.head_dim) # BSD -> BSNH

        qk = nnx.softmax(jnp.einsum("btnh, bsnh -> bnts", q, k) / d**0.5)
        qkv = jnp.einsum("bnts, bsnh -> bnth", qk, v)

        qkv = jnp.transpose(qkv, axes=(0, 2, 1, 3)) # BNTH -> BTNH
        qkv = qkv.reshape(b, t, d)
        out = self.output(qkv)
        return out


class MLP(nnx.Module):
    def __init__(self, input_dim: int, ffw_dim: int, output_dim: int, rngs: nnx.Rngs):
        self.layer_norm = nnx.LayerNorm(epsilon=1e-6, num_features=input_dim, rngs=rngs)
        self.linear1 = nnx.Linear(input_dim, ffw_dim, rngs=rngs)
        self.linear2 = nnx.Linear(ffw_dim, output_dim, rngs=rngs)

    def __call__(self, inputs):
        input_norm = self.layer_norm(inputs)
        x = jax.nn.relu(self.linear1(input_norm))
        out = self.linear2(x)
        return out
        

class MoE(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass


class TokenAndPositionalEmbeddings(nnx.Module):
    def __init__(self, embed_dim: int, rngs: nnx.Rngs, max_len: int=12, vocab_size: int=12):
            self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
            self.pos_emb = nnx.Embed(num_embeddings=max_len, features=embed_dim, rngs=rngs)
    
    def __call__(self, x):
        token_embeddings = self.token_emb(x) 
        positions = jnp.arange(0, x.shape[1])[None, :]
        positional_embeddings = self.pos_emb(positions)
        return token_embeddings + positional_embeddings



# testing
batch = 5
seq_len = 12
emb_dim=4
num_heads=2
ffw_dim=8
output_dim = emb_dim
rngs=nnx.Rngs(0)

test_batch = jnp.ones([batch, seq_len, emb_dim]) # [4, 12, 4]

# transformer layer
transformer_layer = TransformerLayer(emb_dim, num_heads, rngs=rngs)
print(transformer_layer(test_batch).shape)

# mlp layer
mlp = MLP(input_dim=emb_dim, ffw_dim=ffw_dim, output_dim=output_dim, rngs=rngs)
print(mlp(test_batch).shape)

# transformer block (transformer + mlp)
transformer_block = TransformerBlock(emb_dim, num_heads, ffw_dim, False, 0, 0, rngs=rngs)
print(transformer_block(test_batch).shape)

# mlp babychilla 
full_test_batch = jnp.ones([batch, seq_len], dtype=int) # [4, 12] so that the embedding layers work

babychilla_mlp = BabyChilla(embed_dim=emb_dim, num_heads=num_heads, ffw_dim=ffw_dim, layers=2, rngs=rngs)
print(babychilla_mlp(full_test_batch).shape)
print(babychilla_mlp(full_test_batch))

# moe babychilla 
# babychilla_mlp = BabyChilla(embed_dim=emb_dim, num_heads=num_heads, ffw_dim=ffw_dim, layers=2, rngs=rngs, MoE=True)
# print(babychilla_mlp(test_batch).shape)
# print(babychilla_mlp(test_batch))