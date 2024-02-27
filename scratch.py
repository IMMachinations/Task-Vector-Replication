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

def print_test_component_hypothesis_results(results: Tuple[int,int,int,List[int]]) -> None:
    print(f"Baseline hits: {results[1]}/{results[0]}")
    print(f"Regular hits: {results[2]}/{results[0]}")
    print(f"Reconstructed hits: {[result for result in results[3]]}")

# %%
test_component_hypothesis(low_to_caps, arrow, model, num_contexts=1024)
# %%
test_component_hypothesis(caps_to_low, arrow, model, num_contexts=1024)
# %%
test_component_hypothesis(following_number, arrow, model, num_contexts=1024)
# %%
letter_to_low_result = test_component_hypothesis(letter_to_low, arrow, model, num_contexts=2048, len_contexts=6)
letter_to_caps_result = test_component_hypothesis(letter_to_caps, arrow, model, num_contexts=2048, len_contexts=6)
# %%
def substitute_task(taskA: List[Tuple[str,str]], taskB: List[Tuple[str,str]], layer: int, function_token:str = "→", model: HookedTransformer = model, num_contexts: int = 256, len_contexts: int = 4) -> Float:
    ### Check the tasks are the same length and have the same domains
    if(len(taskA) != len(taskB)):
        raise ValueError("The two tasks must have the same length")
    
    taskA.sort(key=lambda x: x[0])
    taskB.sort(key=lambda x: x[0])

    for i in range(len(taskA)):
        if(taskA[i][0] != taskB[i][0]):
            raise ValueError("The two tasks must have the same domains")
    
    ### Pair the tasks together
    mixed_tasks = [(taskA[i][0],taskA[i][1],taskB[i][1]) for i in range(len(taskA))]
    taskA_hits = 0
    taskB_hits = 0
    taskA_corrupted_B_hits = 0
    taskB_corrupted_A_hits = 0

    for _ in tqdm(range(num_contexts)):
        ### Select a random context and query
        random.shuffle(mixed_tasks)
        taskA_context = list(map(lambda x: (x[0],x[1]), mixed_tasks[:len_contexts]))
        taskB_context = list(map(lambda x: (x[0],x[2]), mixed_tasks[:len_contexts]))

        current_query = mixed_tasks[len_contexts][0][0]
        taskA_answer = mixed_tasks[len_contexts][1]
        taskB_answer = mixed_tasks[len_contexts][2]

        logitsA, cacheA = model.run_with_cache(t.tensor(mix_contexts_and_query(taskA_context, current_query, function_token, model)))
        logitsB, cacheB = model.run_with_cache(t.tensor(mix_contexts_and_query(taskB_context, current_query, function_token, model)))

        if(logits_to_next_token(logitsA) == taskA_answer):
            taskA_hits += 1
        if(logits_to_next_token(logitsB) == taskB_answer):
            taskB_hits += 1
        
        layer_cacheA = cacheA[f"blocks.{layer}.hook_resid_pre"].clone()
        layer_cacheB = cacheB[f"blocks.{layer}.hook_resid_pre"].clone()
        layer_cacheA[0,-1,:] = cacheB[f"blocks.{layer}.hook_resid_pre"][0,-1,:]
        layer_cacheB[0,-1,:] = cacheA[f"blocks.{layer}.hook_resid_pre"][0,-1,:]
        
        logitsA_with_B_corruption = model.forward(layer_cacheA, start_at_layer=layer)
        if(logits_to_next_token(logitsA_with_B_corruption) == taskB_answer):
            taskA_corrupted_B_hits += 1
        logitsB_with_A_corruption = model.forward(layer_cacheB, start_at_layer=layer)
        if(logits_to_next_token(logitsB_with_A_corruption) == taskA_answer):
            taskB_corrupted_A_hits += 1
        
    return(num_contexts, taskA_hits, taskB_hits, taskA_corrupted_B_hits, taskB_corrupted_A_hits)

def print_substitute_task_results(results: Tuple[int,int,int,int,int]) -> None:
    print(f"Task A hits: {results[1]}/{results[0]}")
    print(f"Task B hits: {results[2]}/{results[0]}")
    print(f"Task A with B corruption hits: {results[3]}/{results[0]}")
    print(f"Task B with A corruption hits: {results[4]}/{results[0]}")
    
# %%
substitute_task_layer_13 = substitute_task(letter_to_caps, letter_to_low, 13, arrow, model, num_contexts=512, len_contexts=6)
# %%
substitute_task_layer_14 = substitute_task(letter_to_caps, letter_to_low, 14, arrow, model, num_contexts=2048, len_contexts=6)
# %%
### PART 2: Function Vectors in LLMS
def generate_mean_activation(contexts: List[Tuple[str,str]], query: str, function_token: str, model: HookedTransformer = model, num_contexts: int = 1024, len_contexts: int = 4) -> Float:
    
    shuffled_context = contexts.copy()  
    num_total = 0
    activations = t.zeros((model.cfg.n_layers, model.cfg.n_heads, model.cfg.d_model))
    model.cfg.use_attn_results = True
    for _ in tqdm(range(num_contexts)):
        num_total += 1
        random.shuffle(shuffled_context)
        current_context = shuffled_context[:len_contexts]
        current_query = shuffled_context[len_contexts]
        current_answer = shuffled_context[len_contexts][1]

        normal_tokens = t.tensor(mix_contexts_and_query(current_context, current_query[0], function_token, model))
        _, normal_cache = model.run_with_cache(normal_tokens)
        for i in range(model.cfg.n_layers):
            activations[i,:,:] += normal_cache[f"blocks.{i}.attn.attn_results"][0,-1,:,:] # maybe [0,:,-1,:]
    
    return activations/num_contexts

def gather_head_activations_to_layers(mean_head_activations: Tensor) -> Tensor:
    return mean_head_activations.mean(1)

def layer_addition_hook(hook_value: Tensor, hook: HookPoint, vector: Tensor) -> Tensor:
    hook_value[0,-1,:] = hook_value[0,-1,:] + vector
    return hook_value 
    

def apply_layered_vectors_to_zero_shot(layered_vectors: Tensor, contexts: List[Tuple[str,str]], function_token: str, model: HookedTransformer = model) -> Float:
    layer_sums = [0 * model.cfg.n_layers]
    hook_functions = [lambda hook_value, hook : layer_addition_hook(hook_value, hook, vector) for vector in layered_vectors]
    for context in contexts:
        
        tokens = t.tensor([0, model.to_single_token(context[0]), model.to_single_token(function_token)])
        for i in range(model.cfg.n_layers):
            logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{i}.attn.attn_results", hook_functions[i])])
            if(logits_to_next_token(logits) == model.to_single_token(context[1])):
                layer_sums[i] += 1

    return [(1.0 * layer_sum) / len(contexts) for layer_sum in layer_sums]
            
        #print(logits_to_next_token(logits))
        