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
def logits_to_next_token(logits: Tensor, model: HookedTransformer = model) -> int:
    return model.to_string(t.argmax(logits[0,-1,:]))
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
def generate_mean_activation(contexts: List[Tuple[str,str]], function_token: str, seperator_token: str = ",", model: HookedTransformer = model, num_contexts: int = 1024, len_contexts: int = 4) -> Tensor:
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

# %%
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
        base_probability = (t.nn.Softmax(dim=0)(first_logits[0,-1,:]))[model.to_tokens(context[1],prepend_bos=False)[0]].detach()
        for i in range(model.cfg.n_layers):
            logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{i}.hook_attn_out", hook_functions[i])])
            adjusted_probability = (t.nn.Softmax(dim=0)(logits[0,-1,:]))[model.to_tokens(context[1],prepend_bos=False)[0]].detach()
            layer_sums[i] += (adjusted_probability[0] - base_probability[0]).detach()
        
    return layer_sums / len(contexts)

        #print(logits_to_next_token(logits))


# %%
mean_head_activations = generate_mean_activation(letter_to_caps, arrow, num_contexts=2048, len_contexts=6)
# %%
mean_layer_activations = gather_head_activations_to_layers(mean_head_activations)
# %%
layered_accuracy = apply_layered_vectors_to_zero_shot(mean_layer_activations, letter_to_caps, arrow, model)
#px.line(layered_accuracy, title = "Accuracy by layer of adding average activation to zero-shot pythia 2.8b toCaps task")
# %%
layered_adjusted_probability = apply_layered_vectors_to_zero_shot_by_probability(mean_layer_activations, letter_to_caps, arrow, model)
px.line(layered_adjusted_probability.cpu())

# %%
def head_replacement_hook(hook_value: Tensor, hook: hook_points.HookPoint, head:int, vector: Tensor) -> Tensor:
    hook_value[0,:,head,:] = vector
    return hook_value 

def calculate_average_causal_indirect_effect(mean_head_activations: Tensor, scrambled_prompts: List[str], prompt_answers: List[int], model: HookedTransformer = model) -> Tensor:
    if((mean_head_activations.shape[0] != model.cfg.n_layers) or (mean_head_activations.shape[1] != model.cfg.n_heads) or (mean_head_activations.shape[2] != model.cfg.d_model)):
        raise ValueError("Mean head activations must be of shape (n_layers, n_heads, d_model)")    
    if(len(scrambled_prompts) != len(prompt_answers)):
        raise ValueError("Prompt answers must be of the same length as scrambled prompts")
    attn_result_value = model.cfg.use_attn_result
    model.cfg.use_attn_result = True
    causal_indirect_effects = t.zeros((model.cfg.n_layers, model.cfg.n_heads), device=device)
    hook_functions = [[lambda hook_value, hook : head_replacement_hook(hook_value, hook, head, mean_head_activations[layer,head,:]) for head in range(model.cfg.n_heads)] for layer in range(model.cfg.n_layers)]
    
    for prompt, answer in tqdm(zip(scrambled_prompts, prompt_answers)):
        tokens = model.to_tokens(prompt)
        logits = model.forward(tokens)
        answer_probability = (t.nn.Softmax(dim=0))(logits[0,-1,:])[answer].detach()
        for layer in range(model.cfg.n_layers):
            for head in range(model.cfg.n_heads):
                def hook(hook_value, hook):
                    hook_value[0,:,head,:] = mean_head_activations[layer,head,:]
                    return hook_value
                #altered_logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_result", hook_functions[layer][head])])
                altered_logits = model.run_with_hooks(tokens, fwd_hooks=[(f"blocks.{layer}.attn.hook_result", hook)])
                adjusted_probability = (t.nn.Softmax(dim=0))(altered_logits[0,-1,:])[answer].detach()
                #print(f"{layer}, {head}: {adjusted_probability[0]}")
                causal_indirect_effects[layer,head] += (adjusted_probability[0] - answer_probability[0]).detach()
                
    model.cfg.use_attn_result = attn_result_value
    return causal_indirect_effects / len(scrambled_prompts)
    

def generate_shuffled_prompt(contexts: List[Tuple[str,str]], model, function_token:str = ":", seperator_token: str = None) -> Tuple[List[str], List[int]]:
    shuffled_answers = [context[1] for context in contexts[:-1]]
    random.shuffle(shuffled_answers)
    prompt_str = ""
    for i in range(len(contexts)-1):
        prompt_str += contexts[i][0] + function_token + shuffled_answers[i]
        if(seperator_token is not None):
            prompt_str += seperator_token
    prompt_str += contexts[-1][0] + function_token
    answer_token = model.to_tokens(contexts[-1][1], prepend_bos=False).tolist()[0]
    
    return prompt_str, answer_token
    
def generate_shuffled_prompts(contexts: List[Tuple[str,str]], model, num_prompts: int, prompt_length: int, function_token:str = ":", seperator_token: str = None) -> Tuple[List[str], List[int]]:
    if(prompt_length >= len(contexts)):
        raise ValueError("Prompt length must be less than the number of contexts")
    
    shuffled_context = contexts.copy()  
    prompts = []
    answers = []
    for _ in range(num_prompts):
        random.shuffle(shuffled_context)
        prompt, answer = generate_shuffled_prompt(shuffled_context[:prompt_length + 1], model, function_token, seperator_token)
        prompts.append(prompt)
        answers.append(answer)
    return prompts, answers
        
# %%
prompts, answers = generate_shuffled_prompts(letter_to_caps, model, 12, 4, arrow)
# %%
causal_indirect_effect = calculate_average_causal_indirect_effect(mean_head_activations, prompts, answers, model)
# %%
def assemble_task_vector(mean_head_activations: Tensor, causal_indirect_effects: Tensor, layer: int, num_heads) -> Tensor:
    v,i = t.topk(causal_indirect_effects[:layer+1,:].flatten(), num_heads)
    top_heads = np.array(np.unravel_index(i.numpy(), causal_indirect_effects[:layer+1,:].shape)).T
    task_vector = t.zeros((model.cfg.d_model), device=device)
    for head in top_heads:
        task_vector += mean_head_activations[head[0],head[1],:]
# %%
def assemble_end_list_tasks(objects: List[str], num_lists: int, num_elements: int, seperator: str = ",") -> List[Tuple[str,str]]:
    end_list_tasks = []
    for _ in range(num_lists):
        random.shuffle(objects)
        end_list_tasks.append((seperator.join(objects[:num_elements]), objects[num_elements - 1]))
    return end_list_tasks


us_states =  [
    ' Alabama', ' Alaska', ' Arizona', ' Arkansas', ' California', ' Colorado',
    ' Connecticut', ' Delaware', ' Florida', ' Georgia', ' Hawaii', ' Idaho',
    ' Illinois', ' Indiana', ' Iowa', ' Kansas', ' Kentucky', ' Louisiana',
    ' Maine', ' Maryland', ' Massachusetts', ' Michigan', ' Minnesota',
    ' Mississippi', ' Missouri', ' Montana', ' Nebraska', ' Nevada',
    ' New Hampshire', ' New Jersey', ' New Mexico', ' New York',
    ' North Carolina', ' North Dakota', ' Ohio', ' Oklahoma', ' Oregon',
    ' Pennsylvania', ' Rhode Island', ' South Carolina', ' South Dakota',
    ' Tennessee', ' Texas', ' Utah', ' Vermont', ' Virginia', ' Washington',
    ' West Virginia', ' Wisconsin', ' Wyoming'
]

# %%
last_state = assemble_end_list_tasks(us_states, 2000, 5)
average_head = generate_mean_activation(last_state, ":", num_contexts=2048, len_contexts=3, seperator_token="|")

# %%
shuffled_prompts, correct_answers = generate_shuffled_prompts(last_state, model, 12, 3, ":","|")
causal_indirect_effects = calculate_average_causal_indirect_effect(average_head, shuffled_prompts, correct_answers, model)
px.imshow(causal_indirect_effects.cpu().numpy())
# %%
last_state_task_vector = assemble_task_vector(average_head, causal_indirect_effects, 11, 10)
# %%

logits = model.forward(mix_multitoken_contexts_and_query(last_state[:3], query=last_state[3][0], function_token=":", seperator_token="|", model=model))
print(logits_to_next_token(logits, model=model))
# %%
