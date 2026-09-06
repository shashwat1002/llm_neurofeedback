from copy import deepcopy
from pathlib import Path
from collections import defaultdict
import random
import pandas as pd
import torch
from transformers import DynamicCache
from analysis.process_hidden import eval_classify_hiddens, extract_last_representations, extract_assistant_header
from utils import load_yaml, find_tags_indices, safe_dump

ROOT = Path(__file__).resolve().parent

def longest_common_substring(s1, s2):
    """
    Computes the longest common substring between s1 and s2 using dynamic programming.
    Returns a tuple (length, substring) where `length` is the length of the longest common substring
    and `substring` is one example of such longest common substring.
    """
    m, n = len(s1), len(s2)
    # Create a (m+1)x(n+1) table for DP, initialized to 0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    # We'll store the ending index of s1 for one longest substring found so far
    end_index = 0

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i  # update ending index for s1's substring
            else:
                dp[i][j] = 0  # reset if there's no match

    # The longest common substring is from index end_index - max_length to end_index in s1
    longest_sub = s1[end_index - max_length:end_index]
    return max_length, longest_sub


def find_most_similar(target, string_list):
    """
    Finds the string in string_list with the longest common substring when compared to target.
    Returns a tuple: (best_match, match_length, common_substring)
    """
    best_match = None
    best_length = 0
    best_common_sub = ""

    for s in string_list:
        current_length, common_sub = longest_common_substring(target, s)
        # Debug print: Uncomment the next line to see each result
        # print(f"Comparing with '{s}': longest common substring '{common_sub}' of length {current_length}")
        if current_length > best_length:
            best_length = current_length
            best_match = s
            best_common_sub = common_sub

    return best_match, best_length, best_common_sub


def _init_neurofeedback_prompts(labeler, meta_prompt_key: str) -> tuple[dict, list[dict]]:
    """Load prompts and meta_prompts, format score placeholders in meta_prompt."""
    prompt = load_yaml(ROOT / "configs" / "prompts.yml")
    meta_prompt = load_yaml(ROOT / "configs" / "meta_prompts.yml")[meta_prompt_key]

    min_score, max_score = labeler.min_points, labeler.max_points
    score_set = ', '.join([str(x) for x in range(min_score, max_score + 1)])
    meta_prompt[0]['content'] = (
        meta_prompt[0]['content']
        .replace('<SCORE_SET>', score_set)
        .replace('<LOWEST>', str(min_score))
        .replace('<HIGHEST>', str(max_score))
    )
    return prompt, meta_prompt


def generate_ICL_examples(user_prompt, sentences, layers_score, n_examples, labeler, flip_shown_label=False):
    """Generate ICL examples with their associated decoded brain activation scores,"""
    prompts = []
    original_scores = []
    label_scores = []

    for icl_idx in range(n_examples):
        train_example_assistant_response = sentences[icl_idx]
        original_score = layers_score[icl_idx]
        label_score = labeler.transform(original_score)
        if flip_shown_label:
            label_score = labeler.max_points - label_score + labeler.min_points
        label_scores.append(label_score)
        original_scores.append(original_score)
        prompts.append({"role": "user", "content": f"{user_prompt}\n"})
        prompts.append({"role": "assistant", "content": f"{train_example_assistant_response} [Score: {label_score}]\n"})

    return prompts, label_scores, original_scores


def get_choice_scores(logits, tokenizer, choices):
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[-1] for c in choices]
    candidate_logits = logits[:, choice_ids]
    probs = torch.softmax(candidate_logits, dim=-1)
    return probs


@torch.inference_mode()
def imitate_score_by_examples(model, tokenizer, all_classifiers, all_examples_scores, layer, labeler, n_icl_examples,
                              process_hidden_method, save_file, pc_number, scenario='NF', max_new_tokens=50):
    """By showing the LLM ICL examples and their associated decoded brain activation scores,
        ask the LLM to control its brain activation score w/ or w/o generating new responses.
    """
    prompt, meta_prompt = _init_neurofeedback_prompts(labeler, f'control_{scenario}')
    min_score, max_score = labeler.min_points, labeler.max_points

    n_layers = model.config.num_hidden_layers
    assistant_tag_name = extract_assistant_header(tokenizer)
    assistant_tag = (assistant_tag_name, tokenizer.encode(assistant_tag_name, add_special_tokens=False, return_tensors="pt")[0])
    eos_tag_name = tokenizer.special_tokens_map['eos_token']
    eos_tag = (eos_tag_name, tokenizer.encode(eos_tag_name, add_special_tokens=False, return_tensors="pt")[0])
    confidence_score = [i for i in range(1, prompt['confidence_level'] + 1)]

    imitate_score_dt = defaultdict(list)
    examples_assistant_response = all_examples_scores['sentences'].tolist()

    for flip_shown_label in [False, True]:
        prompts, label_scores, original_scores = generate_ICL_examples(
            prompt["user_msg"], examples_assistant_response, all_examples_scores[layer].tolist(),
            n_icl_examples[-1], labeler, flip_shown_label)  # generate max number of ICL examples
        icl_cache = DynamicCache(config=model.config)
        if not flip_shown_label:  # save the original ICL examples and scores
            safe_dump({'original_scores': original_scores, 'label_scores': label_scores,
                       'labeler': labeler}, save_file.with_name(f"original_{save_file.name}"))

        for n in n_icl_examples:  # use first n examples
            current_prompt = deepcopy(meta_prompt + prompts[:2 * n])
            full_prompt = tokenizer.apply_chat_template(current_prompt, tokenize=False, add_generation_prompt=False)
            tokens = tokenizer(full_prompt, return_tensors="pt").to(model.device)
            current_len = tokens['input_ids'].shape[1]
            cache_position = torch.arange(icl_cache.get_seq_length(), current_len, device=model.device)
            inputs = {'input_ids': tokens['input_ids'][:, icl_cache.get_seq_length():], 'attention_mask': tokens['attention_mask']}
            model(**inputs, past_key_values=icl_cache, cache_position=cache_position, use_cache=True)

            # === starting the control task ===
            for i_imit_label, imitate_label in enumerate([min_score, max_score]):
                confidence_check = [
                    {"role": "user", "content": prompt['confidence'].replace('<TARGET_SCORE>', str(imitate_label))},
                    {"role": "assistant", "content": "Confidence: <"},
                ]
                confidence_check = tokenizer.apply_chat_template(
                    confidence_check, tokenize=True, continue_final_message=True, return_tensors="pt").to(model.device)
                for imitate_mode in ['explicit', 'implicit']:
                    eval_prompt = deepcopy(current_prompt)
                    new_instruction = prompt[f'control_{scenario}'].replace('<TARGET_SCORE>', str(imitate_label))
                    eval_prompt.append({"role": "user", "content": new_instruction + prompt["user_msg"]})

                    if imitate_mode == 'explicit':
                        eval_prompt_temp = tokenizer.apply_chat_template(eval_prompt, tokenize=False, add_generation_prompt=True)
                        tokens = tokenizer(eval_prompt_temp, return_tensors="pt").to(model.device)
                        inputs = {'input_ids': tokens['input_ids'][:, icl_cache.get_seq_length():],
                                  'attention_mask': tokens['attention_mask']}
                        cache_position = torch.arange(icl_cache.get_seq_length(), tokens['input_ids'].shape[1], device=model.device)
                        generated_sentence = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=True,
                                                            temperature=0.7, past_key_values=deepcopy(icl_cache),
                                                            cache_position=cache_position, use_cache=True,
                                                            pad_token_id=tokenizer.eos_token_id)
                        generated_sentence = tokenizer.decode(generated_sentence[0][inputs['input_ids'].shape[1]:],
                                                              skip_special_tokens=True)
                        eval_prompt.append({"role": "assistant", "content": f"{generated_sentence}\n"})
                    elif imitate_mode == 'implicit':  # prefill with the last example
                        # eval_prompt.append({"role": "assistant", "content": f"{examples_assistant_response[-1]}\n"})
                        eval_prompt.append({"role": "assistant", "content": f"{random.choice(examples_assistant_response[n:])}\n"})
                    else:
                        raise ValueError(f"Unknown imitation mode: {imitate_mode}")

                    # === extract hiddens ===
                    eval_cache = deepcopy(icl_cache)
                    eval_prompt = tokenizer.apply_chat_template(eval_prompt, tokenize=False)
                    tokens = tokenizer(eval_prompt, return_tensors="pt").to(model.device)  # (batch_size, seq_len)
                    cache_position = torch.arange(eval_cache.get_seq_length(), tokens['input_ids'].shape[1], device=model.device)
                    inputs = {'input_ids': tokens['input_ids'][:, eval_cache.get_seq_length():], 'attention_mask': tokens['attention_mask']}
                    outputs = model(**inputs, output_hidden_states=True, past_key_values=eval_cache, cache_position=cache_position, use_cache=True)
                    hiddens = torch.stack([outputs.hidden_states[j + 1] for j in range(n_layers)], dim=0)  # (n_layers, batch_size, seq_len, hidden_size)
                    inputs['attention_mask'] = tokens['attention_mask'][:, icl_cache.get_seq_length():]  # update attention mask for extraction hiddens, do not use updated eval_cache
                    hiddens = extract_last_representations(hiddens, inputs, assistant_tag, eos_tag, process_hidden_method).cpu()

                    # === estimate the confidence level ===
                    attention_mask = torch.cat([tokens['attention_mask'], tokens['attention_mask'].new_ones((tokens['attention_mask'].shape[0], confidence_check.shape[1]))], dim=-1)
                    inputs = {'input_ids': confidence_check, 'attention_mask': attention_mask}
                    cache_position = torch.arange(eval_cache.get_seq_length(), eval_cache.get_seq_length() + confidence_check.shape[1], device=model.device)
                    logits = model(**inputs, past_key_values=eval_cache, cache_position=cache_position, use_cache=True).logits
                    probs = get_choice_scores(logits[:, -1, :], tokenizer, choices=[str(i) for i in confidence_score]).squeeze().cpu()

                    imitate_score_dt['n_examples'].append(n)
                    imitate_score_dt['imitate_label'].append(i_imit_label)  # use index to represent the highest or lowest score
                    imitate_score_dt['imitate_mode'].append(imitate_mode)
                    imitate_score_dt['flip_shown_label'].append(flip_shown_label)
                    imitate_score_dt['layer'].append(layer)
                    control_score = eval_classify_hiddens(hiddens, all_classifiers, 'score', layer=layer, pc_number=pc_number)
                    imitate_score_dt['imitate_example_scores'].append(control_score.item())
                    imitate_score_dt['processed_hiddens'].append(hiddens.squeeze())
                    imitate_score_dt['confidence'].append(probs)

    imitate_score_dt = pd.DataFrame(imitate_score_dt)
    safe_dump(imitate_score_dt, save_file)


@torch.inference_mode()
def predict_score_by_examples(model, tokenizer, all_examples_scores, labeler, process_hidden_method, save_file,
                              save_indices=None, scenario='NF'):
    """By showing the model examples and their associated decoded brain activation scores,
    can model learn to predict the score given a new example?
    """
    prompt, meta_prompt = _init_neurofeedback_prompts(labeler, f'report_{scenario}')
    min_score, max_score = labeler.min_points, labeler.max_points

    est_score_dt = defaultdict(list)
    layers_scores = {'layer': [], 'original_scores': [], 'labeler': [], 'label_scores': []}
    examples_assistant_response = all_examples_scores['sentences'].tolist()
    n_examples = len(all_examples_scores)
    all_layers = sorted(col for col in all_examples_scores.columns if not isinstance(col, str))
    text_precede_label = "Score: "
    text_precede_label_tokens = tokenizer(text_precede_label, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    possible_choices = [str(i) for i in range(min_score, max_score + 1)]

    for flip_shown_label in [False, True]:
        for layer in all_layers:
            labeler.fit(all_examples_scores[layer].to_numpy())
            prompts, label_scores, original_scores = generate_ICL_examples(
                prompt["user_msg"], examples_assistant_response, all_examples_scores[layer].tolist(),
                n_examples, labeler, flip_shown_label)  # generate max number of ICL examples

            current_prompt = meta_prompt + prompts
            current_prompt = tokenizer.apply_chat_template(current_prompt, tokenize=False)
            tokens = tokenizer(current_prompt, return_tensors="pt").to(model.device)
            outputs = model(**tokens, output_hidden_states=True)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            hiddens = outputs.hidden_states[layer + 1].squeeze()  # (seq_len, hidden_size)
            current_prompt_tokens = tokens['input_ids'][0].detach().cpu()

            # locate text_precede_label_tokens in current_prompt_tokens
            occurrences = find_tags_indices(current_prompt_tokens, [(text_precede_label, text_precede_label_tokens)])[text_precede_label]
            all_example_est_probs = []
            all_example_hiddens = []
            for occ in occurrences:
                start_idx, end_idx = occ # the position for "Score: {", notice that the end_idx is exclusive
                # end_idx -1 is "{", end_idx is the label - "0" or "1"
                # we want to predict the token at end_idx, so logit at end_idx - 1 is the goal
                # print(tokenizer.decode(current_prompt_tokens[start_idx:end_idx]))
                probs = get_choice_scores(logits[:, end_idx-1, :], tokenizer, choices=possible_choices).squeeze()
                all_example_est_probs.append(probs)
                if process_hidden_method == 'mean':
                    all_example_hiddens.append(hiddens[start_idx:end_idx].mean(dim=0))
                elif process_hidden_method == 'last':
                    all_example_hiddens.append(hiddens[end_idx - 1])
                else:
                    raise ValueError(f"Unknown process_hidden_method: {process_hidden_method}")

            est_score_dt['layer'].append(layer)
            est_score_dt['flip_shown_label'].append(flip_shown_label)
            all_example_est_probs = torch.stack(all_example_est_probs, dim=0).cpu()  # (n_examples, n_ratings)
            all_example_est_scores = (all_example_est_probs.argmax(dim=1) + min_score).tolist()
            est_score_dt['all_example_est_scores'].append(all_example_est_scores)
            est_score_dt['all_example_true_scores'].append(label_scores)
            eps = 1e-8
            est_score_dt['all_example_est_scores_logitdiff'].append(
                torch.log((all_example_est_probs[:, -1] + eps) / (all_example_est_probs[:, 0] + eps)).tolist())
            all_example_hiddens = torch.stack(all_example_hiddens, dim=0).cpu()  # (n_examples, hidden_size)
            if save_indices is not None:
                all_example_hiddens = all_example_hiddens[save_indices]
            est_score_dt['processed_hiddens'].append(all_example_hiddens)

            if not flip_shown_label:
                layers_scores['layer'].append(layer)
                layers_scores['original_scores'].append(original_scores)
                layers_scores['label_scores'].append(label_scores)
                layers_scores['labeler'].append(labeler)

    layers_scores = pd.DataFrame(layers_scores)
    safe_dump(layers_scores, save_file.with_name(f"original_{save_file.name}"))
    est_score_dt = pd.DataFrame(est_score_dt)
    safe_dump(est_score_dt, save_file)


@torch.inference_mode()
def predict_score_by_examples_batched(model, tokenizer, all_examples_scores, labeler, process_hidden_method, save_file,
                                      save_indices=None, scenario='NF', batch_size=8, save_hiddens=True):
    """Batched version of predict_score_by_examples.

    Pre-computes prompts for all layers, then runs batched forward passes (batch_size layers at a time)
    instead of one forward pass per layer. Computationally and IO equivalent to predict_score_by_examples.
    """
    prompt, meta_prompt = _init_neurofeedback_prompts(labeler, f'report_{scenario}')
    min_score, max_score = labeler.min_points, labeler.max_points

    est_score_dt = defaultdict(list)
    layers_scores = {'layer': [], 'original_scores': [], 'labeler': [], 'label_scores': []}
    examples_assistant_response = all_examples_scores['sentences'].tolist()
    n_examples = len(all_examples_scores)
    all_layers = sorted(col for col in all_examples_scores.columns if not isinstance(col, str))
    text_precede_label = "Score: "
    text_precede_label_tokens = tokenizer(text_precede_label, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    possible_choices = [str(i) for i in range(min_score, max_score + 1)]

    orig_padding_side = tokenizer.padding_side
    tokenizer.padding_side = 'left'

    try:
        for flip_shown_label in [False, True]:
            # Pre-compute prompts and metadata for all layers before batching
            all_prompt_strs = []
            all_label_scores = []
            all_original_scores = []

            for layer in all_layers:
                labeler.fit(all_examples_scores[layer].to_numpy())
                prompts, label_scores, original_scores = generate_ICL_examples(
                    prompt["user_msg"], examples_assistant_response, all_examples_scores[layer].tolist(),
                    n_examples, labeler, flip_shown_label)
                current_prompt = meta_prompt + prompts
                prompt_str = tokenizer.apply_chat_template(current_prompt, tokenize=False)
                all_prompt_strs.append(prompt_str)
                all_label_scores.append(label_scores)
                all_original_scores.append(original_scores)

            # Process layers in batches
            for batch_start in range(0, len(all_layers), batch_size):
                batch_end = min(batch_start + batch_size, len(all_layers))
                batch_layers = all_layers[batch_start:batch_end]
                batch_prompt_strs = all_prompt_strs[batch_start:batch_end]

                # Tokenize the batch with left-padding so each sequence is right-aligned
                tokens = tokenizer(batch_prompt_strs, return_tensors="pt", padding=True).to(model.device)
                # tokens['input_ids']: (B, max_seq_len), tokens['attention_mask']: (B, max_seq_len)

                outputs = model(**tokens, output_hidden_states=save_hiddens)
                logits = outputs.logits  # (B, max_seq_len, vocab_size)
                logits = logits.cpu()
                torch.cuda.empty_cache()  # Clear GPU cache after moving logits to CPU
                print("model run")

                for i, (layer, layer_idx) in enumerate(zip(batch_layers, range(batch_start, batch_end))):
                    hiddens = outputs.hidden_states[layer + 1][i] if save_hiddens else None  # (max_seq_len, hidden_size)
                    logits_i = logits[i]  # (max_seq_len, vocab_size)

                    # With left padding, real tokens start after the padding tokens
                    n_pad = int((tokens['attention_mask'][i] == 0).sum().item())
                    actual_input_ids = tokens['input_ids'][i][n_pad:].detach().cpu()

                    label_scores = all_label_scores[layer_idx]
                    original_scores = all_original_scores[layer_idx]

                    # Find "Score: " occurrences in the unpadded token sequence
                    occurrences = find_tags_indices(actual_input_ids, [(text_precede_label, text_precede_label_tokens)])[text_precede_label]
                    all_example_est_probs = []
                    all_example_hiddens = []
                    for occ in occurrences:
                        start_idx, end_idx = occ
                        # Offset indices into the padded sequence
                        padded_start = start_idx + n_pad
                        padded_end = end_idx + n_pad
                        probs = get_choice_scores(logits_i[padded_end - 1, :].unsqueeze(0), tokenizer, choices=possible_choices).squeeze()
                        all_example_est_probs.append(probs)
                        if save_hiddens:
                            if process_hidden_method == 'mean':
                                all_example_hiddens.append(hiddens[padded_start:padded_end].mean(dim=0))
                            elif process_hidden_method == 'last':
                                all_example_hiddens.append(hiddens[padded_end - 1])
                            else:
                                raise ValueError(f"Unknown process_hidden_method: {process_hidden_method}")

                    est_score_dt['layer'].append(layer)
                    est_score_dt['flip_shown_label'].append(flip_shown_label)
                    all_example_est_probs = torch.stack(all_example_est_probs, dim=0).cpu()  # (n_examples, n_ratings)
                    all_example_est_scores = (all_example_est_probs.argmax(dim=1) + min_score).tolist()
                    est_score_dt['all_example_est_scores'].append(all_example_est_scores)
                    est_score_dt['all_example_true_scores'].append(label_scores)
                    eps = 1e-8
                    est_score_dt['all_example_est_scores_logitdiff'].append(
                        torch.log((all_example_est_probs[:, -1] + eps) / (all_example_est_probs[:, 0] + eps)).tolist())
                    if save_hiddens:
                        all_example_hiddens = torch.stack(all_example_hiddens, dim=0).cpu()  # (n_examples, hidden_size)
                        if save_indices is not None:
                            all_example_hiddens = all_example_hiddens[save_indices]
                        est_score_dt['processed_hiddens'].append(all_example_hiddens)

                    if not flip_shown_label:
                        layers_scores['layer'].append(layer)
                        layers_scores['original_scores'].append(original_scores)
                        layers_scores['label_scores'].append(label_scores)
                        layers_scores['labeler'].append(labeler)
    finally:
        tokenizer.padding_side = orig_padding_side

    layers_scores = pd.DataFrame(layers_scores)
    safe_dump(layers_scores, save_file.with_name(f"original_{save_file.name}"))
    est_score_dt = pd.DataFrame(est_score_dt)
    safe_dump(est_score_dt, save_file)


@torch.inference_mode()
def predict_score_by_examples_tensor_parallel(model, tokenizer, all_examples_scores, labeler,
                                               process_hidden_method, save_file,
                                               save_indices=None, scenario='NF', save_hiddens=True):
    """Tensor-parallel version of predict_score_by_examples.

    Requires the model to be loaded with tensor parallelism, e.g.:
        import torch.distributed as dist
        dist.init_process_group("nccl")
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        model = AutoModelForCausalLM.from_pretrained(model_name, tp_plan="auto", torch_dtype=..., ...)

    and the script launched via:
        torchrun --nproc_per_node=<n_gpus> script.py

    With tp_plan, every rank holds a shard of each weight matrix; forward passes execute
    in parallel across ranks with an all-reduce at the end of each layer, so each pass
    is faster by ~world_size. All ranks must call this function together.
    Only rank 0 accumulates results and writes to disk.
    """
    import torch.distributed as dist

    is_dist = dist.is_initialized()
    rank = dist.get_rank() if is_dist else 0
    # Each rank owns a slice of every layer; move inputs to its local GPU.
    device = torch.device(f"cuda:{dist.get_rank()}" if is_dist else model.device)

    prompt, meta_prompt = _init_neurofeedback_prompts(labeler, f'report_{scenario}')
    min_score, max_score = labeler.min_points, labeler.max_points

    est_score_dt = defaultdict(list)
    layers_scores = {'layer': [], 'original_scores': [], 'labeler': [], 'label_scores': []}
    examples_assistant_response = all_examples_scores['sentences'].tolist()
    n_examples = len(all_examples_scores)
    all_layers = sorted(col for col in all_examples_scores.columns if not isinstance(col, str))
    text_precede_label = "Score: "
    text_precede_label_tokens = tokenizer(text_precede_label, return_tensors="pt", add_special_tokens=False)['input_ids'][0]
    possible_choices = [str(i) for i in range(min_score, max_score + 1)]

    for flip_shown_label in [False, True]:
        for layer in all_layers:
            labeler.fit(all_examples_scores[layer].to_numpy())
            prompts, label_scores, original_scores = generate_ICL_examples(
                prompt["user_msg"], examples_assistant_response, all_examples_scores[layer].tolist(),
                n_examples, labeler, flip_shown_label)

            current_prompt = meta_prompt + prompts
            current_prompt = tokenizer.apply_chat_template(current_prompt, tokenize=False)
            tokens = tokenizer(current_prompt, return_tensors="pt").to(device)

            # All ranks execute the forward pass cooperatively: each rank computes its weight
            # shard, then an all-reduce assembles the full output.  After the call, every rank
            # has identical logits and hidden_states.
            outputs = model(**tokens, output_hidden_states=save_hiddens)
            logits = outputs.logits  # (1, seq_len, vocab_size)
            hiddens = outputs.hidden_states[layer + 1].squeeze() if save_hiddens else None  # (seq_len, hidden_size)
            current_prompt_tokens = tokens['input_ids'][0].detach().cpu()

            occurrences = find_tags_indices(current_prompt_tokens, [(text_precede_label, text_precede_label_tokens)])[text_precede_label]
            all_example_est_probs = []
            all_example_hiddens = []
            for occ in occurrences:
                start_idx, end_idx = occ
                probs = get_choice_scores(logits[:, end_idx - 1, :], tokenizer, choices=possible_choices).squeeze()
                all_example_est_probs.append(probs)
                if save_hiddens:
                    if process_hidden_method == 'mean':
                        all_example_hiddens.append(hiddens[start_idx:end_idx].mean(dim=0))
                    elif process_hidden_method == 'last':
                        all_example_hiddens.append(hiddens[end_idx - 1])
                    else:
                        raise ValueError(f"Unknown process_hidden_method: {process_hidden_method}")

            # Ensure all ranks finish the forward pass before rank 0 proceeds.
            if is_dist:
                dist.barrier()

            if rank == 0:
                est_score_dt['layer'].append(layer)
                est_score_dt['flip_shown_label'].append(flip_shown_label)
                all_example_est_probs = torch.stack(all_example_est_probs, dim=0).cpu()  # (n_examples, n_ratings)
                all_example_est_scores = (all_example_est_probs.argmax(dim=1) + min_score).tolist()
                est_score_dt['all_example_est_scores'].append(all_example_est_scores)
                est_score_dt['all_example_true_scores'].append(label_scores)
                eps = 1e-8
                est_score_dt['all_example_est_scores_logitdiff'].append(
                    torch.log((all_example_est_probs[:, -1] + eps) / (all_example_est_probs[:, 0] + eps)).tolist())
                if save_hiddens:
                    all_example_hiddens = torch.stack(all_example_hiddens, dim=0).cpu()  # (n_examples, hidden_size)
                    if save_indices is not None:
                        all_example_hiddens = all_example_hiddens[save_indices]
                    est_score_dt['processed_hiddens'].append(all_example_hiddens)

                if not flip_shown_label:
                    layers_scores['layer'].append(layer)
                    layers_scores['original_scores'].append(original_scores)
                    layers_scores['label_scores'].append(label_scores)
                    layers_scores['labeler'].append(labeler)

    if rank == 0:
        layers_scores = pd.DataFrame(layers_scores)
        safe_dump(layers_scores, save_file.with_name(f"original_{save_file.name}"))
        est_score_dt = pd.DataFrame(est_score_dt)
        safe_dump(est_score_dt, save_file)

