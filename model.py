import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

# ffw should equal embed_dim for simplicity

class BabyChilla(nnx.Module):
    def __init__(self, embed_dim: int, ffw_dim: int, num_heads: int, layers: int, rngs: nnx.Rngs, 
                 max_len: int=12, vocab_size: int=12, output_size: int=1998):
        self.token_pos_emb = TokenAndPositionalEmbeddings(embed_dim, rngs)
        self.transformer_blocks = nnx.Sequential(*[TransformerBlock(embed_dim, MoE=False, rngs=rngs) for _ in range(layers)])

    def __call__(self, inputs):
        x = self.token_pos_emb(inputs)
        x = self.transformer_blocks(x)
        out = self.output_layer(x)
        return out

# Transformer blocks made up for Transformer layers followed by MLP or MoE
class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, MoE: bool, rngs: nnx.Rngs, num_experts: int, active_experts: int):
        self.transformer = TransformerLayer(num_heads, embed_dim, rng=rngs)
        
        if not MoE:
            self.ffw = self.MLP(ffw_dim)
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

        self.output = nnx.Linear(in_features=embed_dim, out_features=embed_dim)

    def __call__(self, inputs):
        b, t, d = inputs.shape
        input_norm = self.layer_norm(inputs)
        q = self.q_proj(input_norm)
        k = self.q_proj(input_norm)
        v = self.q_proj(input_norm)

        q = q.reshape(b, t, self.num_heads, self.head_dim) # BTD -> BTNH
        k = k.reshape(b, t, self.num_heads, self.head_dim) # BSD -> BSNH
        v = v.reshape(b, t, self.num_heads, self.head_dim) # BSD -> BSNH

        qk = nnx.softmax((nnx.Einsum("btnh, bsnh -> bnts", q, k) / d**0.5), dim=-1)
        qkv = nnx.Einsum("bnts, bsnh -> bnth", qk, v)

        qkv = qkv.transpose((1, 2)) # BNTH -> BTNH
        out = self.output(qkv)
        return out



class MLP(nnx.Module):
    def __init__(self):
        pass

    def __call__(self, inputs):
        pass

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