import jax
import jax.numpy as jnp
from jax import jit

# Custom tokenizer for my super simple example
# encode each value as it's own token 0-9 
# encode " ":9 (space), "+":10, and "=":11 
# this is a bit aritificial as part of the challenge. Normally i would encode big #s individually
# @jit()

# should " "
operations = {
    " ":9, # should space be -1? This might be a stronger signal for the model
    "+":10,
    "=":11
}

# padded_length = 
def tokenizer(input, padded_length: int = 12):
    """
    padded_length = 12 not 13 because I am removing the space between = and the answer
    """
    tokenized_question = []
    split_input = input.split()
    answer = int(split_input[-1])
    input = " ".join(split_input[:-1])
    for i in range(len(input)):
        if input[i].isdigit():
            tokenized_question.append(int(input[i]))
        else:
            tokenized_question.append(operations[input[i]])
    if len(tokenized_question) < padded_length:
        padding = (padded_length - len(tokenized_question)) * [float("-inf")]
        tokenized_question.extend(padding) 
    return tokenized_question, answer


test = "550 + 43 = 593"

print(tokenizer(test))