import numpy as np
import torch
from tqdm import trange
from analysis.classifiers import PCAClassifier, LogisticRegression, PCAScorer
from typing import List, Union
from utils import extract_assistant_header


def find_last_subseq(tokens, subseq, valid_length):
    """from tokens[-valid_length:] find the last occurrence of subseq"""
    length = len(tokens)
    sub_length = len(subseq)
    for i in range(length - sub_length, length - valid_length - 1, -1):
        if (tokens[i:i + sub_length] == subseq).all():
            return i
    return None


def extract_last_sentence_span(input_ids, valid_length, assistant_ids, end_token_ids):
    last_assistant_ids = find_last_subseq(input_ids, assistant_ids, valid_length)
    if last_assistant_ids is None:
        start = len(input_ids) - valid_length
    else:
        start = last_assistant_ids + len(assistant_ids)

    end = len(input_ids)
    for j in range(start, end):
        if int(input_ids[j].item()) in end_token_ids:
            end = j
            break
    return start, end


def extract_last_representations(hidden, batch_tokens, assistant_tag, eos_tag, method='mean'):
    """Extract the hidden states from last assistant tag to the first eos tag after that"""
    if method == 'mean':
        batch_input_ids = batch_tokens['input_ids'].cpu()
        all_representations = []
        for b in range(batch_input_ids.shape[0]):
            sent_len = int(batch_tokens['attention_mask'][b].sum().item())
            start, end = extract_last_sentence_span(batch_input_ids[b], sent_len, assistant_tag[1], eos_tag[1])
            all_representations.append(hidden[:, b, start:end, :].mean(dim=1).unsqueeze(1))  # (n_layers, 1, hidden_size)
        return torch.cat(all_representations, dim=1)  # (n_layers, batch_size, hidden_size)
    elif method == 'last':
        # assert padding side is left, this is not absolutely correct, but very fast
        return hidden[:, :, -2, :]  # (batch_size, hidden_size), -2 to avoid eos token
    else:
        raise ValueError(f"Unknown method: {method}, must be one of 'mean', 'last'")


@torch.inference_mode()
def get_hiddens(model, tokenizer, data: Union[List[str], str], batch_size=20, method='mean', return_logits=False):
    """Extract hidden states from model for given data
    method determines how to extract the hidden states:
        'mean': average over all tokens from last assistant tag to the first eos tag after that
        'last': use the last token's hidden state
    """
    assert tokenizer.padding_side == 'left', "Tokenizer padding side must be 'left'"
    if isinstance(data, str):
        data = [data]
    logits = []
    hiddens = []

    assistant_tag_name = extract_assistant_header(tokenizer)
    assistant_tag = (assistant_tag_name, tokenizer.encode(assistant_tag_name, add_special_tokens=False, return_tensors="pt")[0])
    eos_tag_name = tokenizer.special_tokens_map['eos_token']
    eos_tag = (eos_tag_name, tokenizer.encode(eos_tag_name, add_special_tokens=False, return_tensors="pt")[0])

    n_layers = model.config.num_hidden_layers
    tokens = tokenizer(data, return_tensors="pt", padding=True).to(model.device)  # (batch_size, seq_len)
    _, seq_len = tokens['input_ids'].shape
    assert seq_len <= model.config.max_position_embeddings, f"Sequence length {seq_len} exceeds model max context length {model.config.max_position_embeddings}"

    for i in trange(0, len(data), batch_size, desc="Extracting hiddens"):
        batch_tokens = {key: value[i:i + batch_size] for key, value in tokens.items()}
        outputs = model(**batch_tokens, output_hidden_states=True)
        if return_logits:
            logits.append(outputs.logits.detach().cpu())
        batch_hiddens = torch.stack([outputs.hidden_states[j+1] for j in range(n_layers)], dim=0)  # (n_layers, batch_size, seq_len, hidden_size)
        batch_hiddens = extract_last_representations(batch_hiddens, batch_tokens, assistant_tag, eos_tag, method)
        hiddens.append(batch_hiddens.detach().cpu())

    hiddens = torch.cat(hiddens, dim=1)  # (n_layers, total_batch_size, hidden_size)
    logits = torch.cat(logits, dim=0) if return_logits else None
    return logits, hiddens


def train_classify_hiddens(hiddens, labels, method='lr', normalize=True, pc_number=None):
    """Train classifiers for each layer using hidden states
    :param hiddens: dict of tensors [batch_size, hidden_size]
    :param labels: list of labels, [batch_size]
    :param method: str, classifier type, 'lr' for logistic regression, 'pca' for principal component analysis
    :param normalize: bool, whether to normalize the hidden states
    :param pc_number: int, number of principal components to use
    :return: dict of classifiers, key: layer, value: sklearn classifier
    """
    all_classifiers = {}
    all_accuracies = {}

    for layer in trange(hiddens.shape[0], desc="Training classifiers"):
        value = hiddens[layer].numpy()
        if method == 'lr':
            clf = LogisticRegression(normalize, max_iter=1000, solver='saga', n_jobs=-1)
        elif method == 'pcadiff':
            clf = PCAClassifier(normalize)
        elif method == 'pcascore':
            clf = PCAScorer(pc_number, normalize)
        else:
            raise ValueError(f"Unknown method: {method}, must be one of 'lr', 'pcadiff', 'pcascore'")
        clf.fit(value, labels)
        all_classifiers[layer] = clf
        all_accuracies[layer] = clf.score(value, labels)

    return all_classifiers, all_accuracies


def eval_hiddens_score(layer_hiddens, clf, return_type='accuracy', labels=None, pc_number=None):
    if pc_number is not None:
        clf.pc_number = pc_number
    if return_type == 'accuracy':
        score = clf.score(layer_hiddens, labels)
    elif return_type == 'score':
        score = clf.decision_function(layer_hiddens)
    else:
        raise ValueError(f"Unknown return_type: {return_type}, must be one of 'accuracy', 'score'")
    return score


def eval_classify_hiddens(hiddens, all_classifiers, return_type='accuracy', layer=None, labels=None, pc_number=None):
    if layer is not None:
        return eval_hiddens_score(hiddens[layer], all_classifiers[layer], return_type, labels, pc_number)
    else:
        all_scores = {}
        keys = hiddens.keys() if isinstance(hiddens, dict) else range(hiddens.shape[0])
        for i in keys:
            all_scores[i] = eval_hiddens_score(hiddens[i], all_classifiers[i], return_type, labels, pc_number)
        return all_scores


@torch.inference_mode()
def decode_hiddens_score(model, tokenizer, text: str, clfs, layers, method='lr'):
    logits, hiddens = get_hiddens(model, tokenizer, text, method='mean')
    scores = []

    if isinstance(layers, int):
        return clfs.decision_function(hiddens[layers])[0]

    for layer in layers:
        if method in ['lr', 'pca']:
            score = clfs[layer].decision_function(hiddens[layer])[0]
        # elif method == 'pca':
        #     # score = project_hiddens(hiddens, clfs['directions'], clfs['means'])
        else:
            raise ValueError(f"Unknown method: {method}, must be one of 'lr', 'pca")
        scores.append(score)

    return np.mean(scores)
