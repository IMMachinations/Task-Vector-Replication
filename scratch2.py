#%%
import torch as t
from torch import nn, Tensor
from torch.nn import functional as F
from dataclasses import dataclass
import time
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import einops
import plotly.express as px
from pathlib import Path
from jaxtyping import Float
from typing import Optional, Callable, Union
from tqdm.auto import tqdm
from dataclasses import dataclass
from transformer_lens import HookedTransformer, hook_points
from typing import List, Tuple
import random



device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
### First, we need a model. 
model = HookedTransformer.from_pretrained("pythia-410m", device=device)
# %%
low_to_caps = [(letter, letter.upper()) for letter in "abcdefghijklmnopqrstuvwxyz"] 
caps_to_low =  [(letter.upper(), letter) for letter in "abcdefghijklmnopqrstuvwxyz"] 
letter_to_caps = [(letter, letter.upper()) for letter in "abcdefghijklmnopqrstuvwxyz"] + [(letter.upper(), letter.upper()) for letter in "abcdefghijklmnopqrstuvwxyz"]
letter_to_low =  [(letter.upper(), letter) for letter in "abcdefghijklmnopqrstuvwxyz"] + [(letter, letter) for letter in "abcdefghijklmnopqrstuvwxyz"]

fruit_to_color = [("apple", "red"), ("banana", "yellow"), ("orange", "orange"),
                   ("strawberry", "red"), ("blueberry", "blue"), ("kiwi", "green"), ("watermelon", "green"), 
                   ("pineapple", "yellow"), ("mango", "orange"), ("peach", "orange"), ("pear", "green"), 
                   ("plum", "purple"), ("cherry", "red"), ("raspberry", "red"), ("blackberry", "black"), 
                   ("cantaloupe", "orange"), ("honeydew", "green"), ("papaya", "orange"), 
                   ("apricot", "orange"), ("nectarine", "orange"), 
                   ("lemon", "yellow"), ("lime", "green"), ("grapefruit", "orange"), ("coconut", "white"),
                     ("pomegranate", "red"), ("fig", "purple"), ("date", "brown")]
following_number = [("one","two"), ("two", "three"),("three", "four"),("four", "five"),("five", "six"),("six", "seven"),("seven", "eight"),("eight", "nine"),("nine", "ten")]

# %%
arrow = "→"
def construct_context(pair: Tuple[str, str], function_token: str = "→") -> str:
    return pair[0] + function_token + pair[1]
def construct_query(pair: Tuple[str, str], function_token: str = "→") -> Tuple[str,str]:
    return (pair[0] + function_token, pair[1])
def mix_contexts_and_query(contexts : List[Tuple[str,str]], query: str, function_token: str = "→", seperator_token: str = None, model: HookedTransformer = model) -> List[int]:
    function_token_int = model.to_single_token(function_token)
    token_list = [0]

    for context in contexts:
        token_list.append(model.to_single_token(context[0]))
        token_list.append(function_token_int)
        token_list.append(model.to_single_token(context[1]))
        if(seperator_token is not None):
            token_list.append(model.to_single_token(seperator_token))
    if(seperator_token is not None):
        token_list.append(model.to_single_token(seperator_token))
    return token_list + [model.to_single_token(query), model.to_single_token(function_token)]
def mix_multitoken_contexts_and_query(contexts : List[Tuple[str,str]], query: str, function_token: str = "→", seperator_token: str = None, model: HookedTransformer = model) -> List[int]:
    function_token_list = model.to_tokens(function_token,prepend_bos=False).tolist()[0]
    token_list = [0]
    has_seperator = seperator_token is not None
    if(has_seperator):
        seperator_tokens = model.to_tokens(seperator_token,prepend_bos=False).tolist()[0]

    for context in contexts:
        token_list += model.to_tokens(context[0], prepend_bos=False).tolist()[0]
        token_list += function_token_list
        token_list += model.to_tokens(context[1], prepend_bos=False).tolist()[0]
        if(has_seperator):
            token_list += seperator_tokens
    if(has_seperator):
        token_list += seperator_tokens
    return token_list + model.to_tokens(query,prepend_bos=False).tolist()[0] + function_token_list
### PART 2: Function Vectors in LLMS
def generate_mean_activation(contexts: List[Tuple[str,str]], function_token: str, seperator_token: str = ",", model: HookedTransformer = model, num_contexts: int = 1024, len_contexts: int = 4) -> Float:
    shuffled_context = contexts.copy()  
    num_total = 0
    activations = t.zeros((model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model), device=device)
    previous_attn_result_value = model.cfg.use_attn_result 
    model.cfg.use_attn_result = True
    for _ in tqdm(range(num_contexts)):
        num_total += 1
        random.shuffle(shuffled_context)
        current_context = shuffled_context[:len_contexts]
        current_query = shuffled_context[len_contexts]
        current_answer = shuffled_context[len_contexts][1]

        normal_tokens = mix_multitoken_contexts_and_query(current_context, current_query[0], function_token, seperator_token, model)
        normal_tokens = t.tensor(normal_tokens)
        _, normal_cache = model.run_with_cache(normal_tokens)
        for i in range(model.cfg.n_layers):
            activations[i,:,:] += normal_cache[f"blocks.{i}.attn.hook_result"][0,-1,:,:] # maybe [0,:,-1,:]
    model.cfg.use_attn_result = previous_attn_result_value
    return activations/num_contexts

def gather_head_activations_to_layers(mean_head_activations: Tensor) -> Tensor:
    return mean_head_activations.sum(1)

def layer_addition_hook(hook_value: Tensor, hook: hook_points.HookPoint, vector: Tensor) -> Tensor:
    hook_value[0,-1,:] = hook_value[0,-1,:] + vector
    return hook_value 

def logits_to_next_token(logits: Tensor, model: HookedTransformer = model) -> int:
    return model.to_string(t.argmax(logits[0,-1,:]))

def apply_layered_vectors_to_zero_shot(layered_vectors: Tensor, contexts: List[Tuple[str,str]], function_token: str, model: HookedTransformer = model) -> Float:
    layer_sums = [0] * model.cfg.n_layers
    #print(layer_sums)
    hook_functions = [lambda hook_value, hook : layer_addition_hook(hook_value, hook, vector) for vector in layered_vectors]
    for context in tqdm(contexts):
        #print(context[1])
        #print(model.to_single_token(context[1])
        tokens = t.tensor([0, model.to_single_token(context[0]), model.to_single_token(function_token)])
        for i in range(model.cfg.n_layers):
            logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{i}.hook_attn_out", hook_functions[i])])
            if(logits_to_next_token(logits) == context[1]):
                layer_sums[i] += 1

    return [1.0 * layer_sum / len(contexts) for layer_sum in layer_sums]

        #print(logits_to_next_token(logits))

# %%
def identify_probability_of_token(logits: Tensor, token: str, model: HookedTransformer = model) -> Float:
    return t.nn.functional.softmax(logits[0,-1,:],dim=0)[model.to_single_token(token)]

def apply_layered_vectors_to_zero_shot_by_probability(layered_vectors: Tensor, contexts: List[Tuple[str,str]], function_token: str, model: HookedTransformer = model) -> Float:
    layer_sums = t.zeros((model.cfg.n_layers), device=device)
    #print(layer_sums)
    hook_functions = [lambda hook_value, hook : layer_addition_hook(hook_value, hook, vector) for vector in layered_vectors]
    first_logits = None
    logits = None
    for context in tqdm(contexts):
        tokens = t.tensor([0] + model.to_tokens(context[0], prepend_bos=False).tolist()[0] +  [model.to_single_token(function_token)], device=device)
        first_logits = model.forward(tokens)
        base_probability = (t.nn.Softmax(dim=0)(first_logits[0,-1,:]))[model.to_tokens(context[1],prepend_bos=False)[0]]
        for i in range(model.cfg.n_layers):
            logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{i}.hook_attn_out", hook_functions[i])])
            adjusted_probability = (t.nn.Softmax(dim=0)(logits[0,-1,:]))[model.to_tokens(context[1],prepend_bos=False)[0]]
            layer_sums[i] += (adjusted_probability[0] - base_probability[0])
        #break
        print(layer_sums.shape)

    return layer_sums / len(contexts)

        #print(logits_to_next_token(logits))


# %%
mean_head_activations = generate_mean_activation(letter_to_caps, arrow, num_contexts=2048, len_contexts=6)
# %%
mean_layer_activations = gather_head_activations_to_layers(mean_head_activations)
# %%
layered_accuracy = apply_layered_vectors_to_zero_shot(mean_layer_activations, letter_to_caps, arrow, model)
px.line(layered_accuracy)
# %%
#layered_adjusted_probability = apply_layered_vectors_to_zero_shot_by_probability(mean_layer_activations, letter_to_caps, arrow, model)
# %%
context = letter_to_caps[0]
tokens = t.tensor([0, model.to_single_token(context[0]), model.to_single_token(arrow)])
logits = model.forward(tokens)
base_probability = identify_probability_of_token(logits, context[1], model)
# %%
