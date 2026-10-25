import jax
import jax.numpy as jnp

import flax.nnx as nnx
import optax

# ffw should equal embed_dim for simplicity

class BabyChillaModel(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, layers: int, rngs: nnx.Rngs, 
                 MoE: bool= False, alpha=1e-3, max_len: int=11, vocab_size: int=14, output_size: int=1999, 
                 num_experts: int=12, active_experts: int=4):
        self.token_pos_emb = TokenAndPositionalEmbeddings(embed_dim, rngs=rngs, max_len=max_len, vocab_size=vocab_size)
        self.transformer_blocks = nnx.Sequential(*[TransformerBlock(embed_dim=embed_dim,
                                                                    num_heads=num_heads,
                                                                    ffw_dim=ffw_dim, 
                                                                    is_MoE=MoE,
                                                                    alpha=alpha,
                                                                    num_experts=num_experts,
                                                                    active_experts=active_experts,
                                                                    rngs=rngs) 
                                                                    for _ in range(layers)])
        self.final_norm = nnx.LayerNorm(embed_dim, rngs=rngs)
        self.output_layer = nnx.Linear(in_features=embed_dim, out_features=output_size, rngs=rngs)

    def __call__(self, x):
        # print(x)
        x = self.token_pos_emb(x)
        # print(f"embeddings: {x}")
        x = self.transformer_blocks(x)
        # retrieve only one of the output points since we are doing classification instead of next token prediction
        # so the output is [num_batches, num_classes]
        #print(f"post transformer blocks {x}")
        x = x[:, 0] # idea from https://flax.readthedocs.io/en/stable/examples/vit_training.html
        out = self.output_layer(x)
        return out

    def get_num_params(self, exclude_embedding=True):
        def count(module):
            return sum(x.size for x in jax.tree.leaves(nnx.state(module, nnx.Param)))

        n_params = count(self)
        if exclude_embedding:
            n_params -= count(self.token_pos_emb)
        return n_params

# Transformer blocks made up for Transformer layers followed by MLP or MoE
class TransformerBlock(nnx.Module):
    def __init__(self, embed_dim: int, num_heads: int, ffw_dim: int, is_MoE: bool, alpha: int, num_experts: int, active_experts: int, rngs: nnx.Rngs):
        self.transformer = TransformerLayer(embed_dim, num_heads, rngs=rngs)
        
        if not is_MoE:
            self.ffw = MLP(input_dim=embed_dim, ffw_dim=ffw_dim, output_dim=embed_dim, rngs=rngs)
        else:
            self.ffw = MoE(alpha=alpha, input_dim=embed_dim, ffw_dim=ffw_dim, output_dim=embed_dim, num_experts=num_experts, active_experts=active_experts, rngs=rngs)

    def __call__(self, inputs):
        #print(inputs)
        transformer_output = self.transformer(inputs) + inputs # residual connection
        #print(f"Transformer output: {transformer_output}")
        ffw_output = self.ffw(transformer_output) + transformer_output # ffw residual connection? 
        #print(f"ffw output: {ffw_output}")
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
        k = self.k_proj(input_norm)
        v = self.v_proj(input_norm)

        q = q.reshape(b, t, self.num_heads, self.head_dim) # BTD -> BTNH
        k = k.reshape(b, t, self.num_heads, self.head_dim) # BSD -> BSNH
        v = v.reshape(b, t, self.num_heads, self.head_dim) # BSD -> BSNH

        qk = nnx.softmax(jnp.einsum("btnh, bsnh -> bnts", q, k) / k.shape[-1]**0.5, axis=-1)
        qkv = jnp.einsum("bnts, bsnh -> bnth", qk, v)

        qkv = jnp.transpose(qkv, axes=(0, 2, 1, 3)) # BNTH -> BTNH
        qkv = qkv.reshape(b, t, d) # BTNH -> BTD
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
    def __init__(self, alpha, input_dim, ffw_dim, output_dim, num_experts, active_experts, rngs):
        self.alpha = alpha
        self.num_experts = num_experts
        self.active_experts = active_experts

        self.input_layer_norm = nnx.LayerNorm(epsilon=1e-6, num_features=input_dim, rngs=rngs)
        self.output_layer_norm = nnx.LayerNorm(epsilon=1e-6, num_features=output_dim, rngs=rngs)
        self.router = nnx.Linear(input_dim, num_experts, rngs=rngs)

        self.ffn_up = nnx.Param(nnx.initializers.lecun_normal(batch_axis=0)(rngs.params(), (num_experts, input_dim, ffw_dim)))
        self.ffn_out = nnx.Param(nnx.initializers.lecun_normal(batch_axis=0)(rngs.params(), (num_experts, ffw_dim, output_dim)))

    def __call__(self, inputs):
        batch, seq_len, emb_dim = inputs.shape
        input_norm = self.input_layer_norm(inputs) # pre-norm is generally standard and agrees with 

        # this needs to be by token not by sequence
        # input size is [batch, seq_len, emb_dim]
        # want to combine batch & seq_len to get [tokens, emb_dim]
        # pass through router so we get [tokens, num_experts] and softmax
        # Take top_k here then sort into a new array
        # Pass into jax.lax.ragged_dot(lhs, rhs, group_sizes)
        # ragged_up
        # lhs [tokens, emb_dim] This is my input
        # rhs [Experts, emb_dim, ffw_dim] Params
        # ragged_down
        # lhs [tokens, ffw_dim] This is my input
        # rhs [Experts, ffw_dim, output_dim] Params
        # Where group_sizes is the number of tokens per Expert
        
        tokens = input_norm.reshape(batch*seq_len, emb_dim) # reshape to [tokens, emb_dim]
        routed_inputs = nnx.softmax(self.router(tokens), axis=-1) # should return [tokens, num_experts] softmax
        top_k_vals, top_k_indices  = jax.lax.top_k(routed_inputs, self.active_experts, axis=-1) # returns [values, indices]

        # I am sure this isn't jax best practice but it scales to multiple experts for now
        experts = [[i, expert_num] for i, experts in enumerate(top_k_indices) for expert_num in experts]
        sorted_experts = sorted(experts, key=lambda x: x[1])
        expert_order = jnp.array([x[0] for x in sorted_experts]) # this should be an array of indexes 
        expert_counts = jnp.array([x[1] for x in sorted_experts]) # this is an array of which expert each token uses. Here to calculate group sizes

        tokens_ordered_by_expert = tokens[expert_order] # [tokens, emb_dim] ordered by their expert
        #print(tokens_ordered_by_expert)
        group_sizes = jnp.bincount(expert_counts, length=self.num_experts)

        # jax.lax.ragged_dot a meme at this point
        ragged_up = jax.lax.ragged_dot(tokens_ordered_by_expert, self.ffn_up.value, group_sizes)
        ragged_up = nnx.relu(ragged_up)
        ragged_out = jax.lax.ragged_dot(ragged_up, self.ffn_out.value, group_sizes)
        print(ragged_out)

        # reorder and reshape
        expert_tokens = jnp.zeros_like(ragged_out) # [0]*len(ragged_out)
        for i, expert_idx in enumerate(expert_order):
            expert_tokens = expert_tokens.at[i].set(ragged_out[expert_idx])

        # Sum and average across experts
        expert_tokens = expert_tokens.reshape(batch, seq_len, self.active_experts, emb_dim)
        summed_expert_tokens = expert_tokens.sum(axis=2)
        normalized_expert_tokens = self.output_layer_norm(summed_expert_tokens)

        output = normalized_expert_tokens.reshape(batch, seq_len, emb_dim)

        # get auxiliary loss
        aux_loss = self.auxiliary_loss(alpha=self.alpha, N=self.num_experts, K=self.active_experts, routed_inputs=routed_inputs, top_k_indices=top_k_indices)

        # return
        return output, aux_loss

    def auxiliary_loss(self, alpha, N, K, routed_inputs, top_k_indices):
        T = len(routed_inputs)

        def f(N):
            vals = []
            for i in range(N):
                total = 0
                for t in range(T):
                    if i in top_k_indices[t]:
                        total += 1
                vals.append(N/(K*T)*total)
            return jnp.array(vals)

        def P(N):
            vals = []
            for i in range(N):
                total = 0
                for t in range(T):
                    total += routed_inputs[t][i]
                vals.append((1/T) * total)
            return jnp.array(vals)

        total_loss = alpha * (f(N) @ P(N))

        return total_loss



class TokenAndPositionalEmbeddings(nnx.Module):
    def __init__(self, embed_dim: int, rngs: nnx.Rngs, max_len: int=11, vocab_size: int=12):
            self.token_emb = nnx.Embed(num_embeddings=vocab_size, features=embed_dim, rngs=rngs)
            self.pos_emb = nnx.Embed(num_embeddings=max_len, features=embed_dim, rngs=rngs)
    
    def __call__(self, x):
        token_embeddings = self.token_emb(x) 
        positions = jnp.arange(0, x.shape[1])[None, :]
        positional_embeddings = self.pos_emb(positions)
        return token_embeddings + positional_embeddings






# # testing
# batch = 5
# seq_len = 12
# emb_dim=4
# num_heads=2
# ffw_dim=8
# output_dim = emb_dim
# rngs=nnx.Rngs(0)

# test_batch = jnp.ones([batch, seq_len, emb_dim]) # [4, 12, 4]

# # transformer layer
# transformer_layer = TransformerLayer(emb_dim, num_heads, rngs=rngs)
# print(transformer_layer(test_batch).shape)

# # mlp layer
# mlp = MLP(input_dim=emb_dim, ffw_dim=ffw_dim, output_dim=output_dim, rngs=rngs)
# print(mlp(test_batch).shape)

# # transformer block (transformer + mlp)
# transformer_block = TransformerBlock(emb_dim, num_heads, ffw_dim, False, 0, 0, rngs=rngs)
# print(transformer_block(test_batch).shape)

# # mlp babychilla 
# full_test_batch = jnp.ones([batch, seq_len], dtype=int) # [4, 12] so that the embedding layers work

# babychilla_mlp = BabyChillaModel(embed_dim=emb_dim, num_heads=num_heads, ffw_dim=ffw_dim, layers=2, rngs=rngs)
# print(babychilla_mlp(full_test_batch).shape)
# print(babychilla_mlp(full_test_batch))

# moe babychilla 
# babychilla_mlp = BabyChillaModel(embed_dim=emb_dim, num_heads=num_heads, ffw_dim=ffw_dim, layers=2, rngs=rngs, MoE=True)
# print(babychilla_mlp(test_batch).shape)
# print(babychilla_mlp(test_batch))