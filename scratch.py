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
from transformer_lens import HookedTransformer
from typing import List, Tuple
import random



device = t.device("cuda" if t.cuda.is_available() else "cpu")
# %%
### First, we need a model. 
model = HookedTransformer.from_pretrained("pythia-410m")
# %%
low_to_caps = [(letter, letter.upper()) for letter in "abcdefghijklmnopqrstuvwxyz"]
caps_to_low = [(letter.upper(), letter) for letter in "abcdefghijklmnopqrstuvwxyz"]
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
def mix_contexts_and_query(contexts : List[Tuple[str,str]], query: str, function_token: str = "→", model: HookedTransformer = model) -> List[int]:
    function_token_int = model.to_single_token(function_token)
    token_list = []

    for context in contexts:
        token_list.append(model.to_single_token(context[0]))
        token_list.append(function_token_int)
        token_list.append(model.to_single_token(context[1]))
    return token_list + [model.to_single_token(query), model.to_single_token(function_token)]
# %%
low_to_caps_strings = [construct_context(pair, arrow) for pair in low_to_caps]
low_to_caps_queries = [construct_query(pair, arrow) for pair in low_to_caps]
# %%
tokens = model.to_tokens(low_to_caps_strings[0])
print(tokens)
# %%
tokens = t.cat((model.to_tokens(low_to_caps_strings[0],prepend_bos=True) 
          , model.to_tokens(low_to_caps_strings[3],prepend_bos=False) 
          , model.to_tokens(low_to_caps_strings[7],prepend_bos=False) 
          , model.to_tokens(low_to_caps_queries[5][0],prepend_bos=False)), 1)

print(tokens)
print(model.to_string(model.generate(tokens,1)))
# %%
#print(mix_contexts_and_query(low_to_caps[:3],low_to_caps[4][0], arrow, model))
#print(model(t.tensor(mix_contexts_and_query(low_to_caps[:3],low_to_caps[4][0], arrow, model))))
#print(t.tensor(mix_contexts_and_query(low_to_caps[:3],low_to_caps[4][0], arrow, model)))
#print(model.run_with_cache(t.tensor(mix_contexts_and_query(low_to_caps[:4],low_to_caps[7][0], arrow, model))))
print(model.run_with_cache(t.tensor(mix_contexts_and_query(low_to_caps[:4],low_to_caps[7][0], arrow, model)))[0][0,-1,:].shape)
top_10_logits_index = t.topk(model.run_with_cache(t.tensor(mix_contexts_and_query(low_to_caps[:4],low_to_caps[7][0], arrow, model)))[0][0,-1,:],10).indices.tolist()#.topk(10).indices
print(model.to_string(top_10_logits_index))
# %%
def logits_to_next_token(logits: Tensor, model: HookedTransformer = model) -> int:
    return model.to_string(t.argmax(logits[0,-1,:]))
print(logits_to_next_token(model.run_with_cache(t.tensor(mix_contexts_and_query(low_to_caps[:4],low_to_caps[7][0], arrow, model)))[0]))
# %%
def test_component_hypothesis(contexts: List[Tuple[str,str]], function_token: str, model: HookedTransformer = model, num_contexts: int = 256, len_contexts: int = 4) -> Float:
    
    shuffled_context = contexts.copy()  

    num_total = 0
    num_baseline = 0
    num_regular = 0
    num_reconstructed = [0 for _ in range(model.cfg.n_layers)]
    

    for _ in tqdm(range(num_contexts)):
        num_total += 1
        ### Select the context, query, and dummy
        random.shuffle(shuffled_context)
        current_context = shuffled_context[:len_contexts]
        current_query = shuffled_context[len_contexts]
        current_answer = shuffled_context[len_contexts][1]
        current_dummy = shuffled_context[len_contexts+1][0]

        ### Run the baseline
        baseline_tokens = t.tensor(model.to_tokens(construct_query(current_query, function_token)[0]))
        if(logits_to_next_token(model(baseline_tokens)) == current_answer):
            num_baseline += 1

        ### Run Normal with cache
        normal_tokens = t.tensor(mix_contexts_and_query(current_context, current_query[0], function_token, model))
        normal_logits, normal_cache = model.run_with_cache(normal_tokens)
        if(logits_to_next_token(normal_logits) == current_answer):
            num_regular += 1

        ### Run dummy token with cache
        _, dummy_cache = model.run_with_cache(t.tensor(mix_contexts_and_query(current_context, current_dummy, function_token, model)))
        
        ### Run normal token with dummy cache patched in to the -1 position for each layer
        for layer in range(model.cfg.n_layers):
            layer_cache = dummy_cache[f"blocks.{layer}.hook_resid_pre"]#.clone()
            layer_cache[0,-2,:] = normal_cache[f"blocks.{layer}.hook_resid_pre"][0,-2,:]
            model_from_layer = model.forward(layer_cache,start_at_layer=layer)
            if(logits_to_next_token(model_from_layer) == current_answer):
                num_reconstructed[layer] += 1

    return (num_total, num_baseline, num_regular, num_reconstructed)


# %%
test_component_hypothesis(low_to_caps, arrow, model, num_contexts=1024)
# %%
test_component_hypothesis(caps_to_low, arrow, model, num_contexts=1024)
# %%
test_component_hypothesis(following_number, arrow, model, num_contexts=1024)
# %%

# %%
