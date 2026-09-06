import os
import random
import json
from glob import glob
import pandas as pd
from utils import seed_everything


def _compute_effective_ratio(n_sample, used_examples, train_ratio, min_split=0):
    """Compute effective train ratio when n_sample may exceed used_examples."""
    if used_examples == 0:
        return train_ratio
    base = n_sample if n_sample is not None else used_examples
    split_raw = int(base * train_ratio)
    if min_split > 0:
        split_capped = min(max(split_raw, min_split), used_examples - min_split)
    else:
        split_capped = max(0, min(split_raw, used_examples))
    return split_capped / used_examples


def _interleave_balanced(df, label_col='label'):
    """Interleave rows with label 0 and 1 to create balanced dataset."""
    df_0 = df[df[label_col] == 0]
    df_1 = df[df[label_col] == 1]
    shorter_len = min(len(df_0), len(df_1))
    interleaved = [row for pair in zip(df_0[:shorter_len].iterrows(), df_1[:shorter_len].iterrows()) for row in pair]
    return pd.DataFrame([row[1] for row in interleaved])


def _split_examples(examples, train_ratio):
    def pack(split_examples):
        assert len(split_examples[0]) == 2, "Each example must be a tuple of length 2."
        return {"sentences": [item[0] for item in split_examples], "labels": [item[1] for item in split_examples]}
    split_index = int(len(examples) * train_ratio)
    split_index = max(0, min(split_index, len(examples)))
    train_examples = examples[:split_index]
    test_examples = examples[split_index:]
    print(f"Total examples: {len(examples)}")
    print(f"Train samples: {len(train_examples)}")
    print(f"Test samples: {len(test_examples)}")
    return {"train": pack(train_examples), "test": pack(test_examples)}


def honesty_function_dataset(data_path, tags, shuffle=False, n_train=512, include_tf='true'):
    # recommend to use include_tf='true' for the honesty function dataset
    # to simplify the task for presentation
    assert include_tf in ['true', 'false', 'both']
    user_tag = tags['user']
    assistant_tag = tags['assistant']
    df = pd.read_csv(data_path)
    true_statements = df[df['label'] == 1]['statement'].tolist()
    false_statements = df[df['label'] == 0]['statement'].tolist()

    template = "Pretend you're {type} person making statements about the world."
    combined = []

    if include_tf in ['true', 'both']:
        for stmt in true_statements:
            honest = f"{user_tag} {template.format(type='an honest')} {assistant_tag} {stmt}".strip()
            dishonest = f"{user_tag} {template.format(type='a dishonest')} {assistant_tag} {stmt}".strip()
            combined.append((honest, dishonest, 1))  # 1 for true

    if include_tf in ['false', 'both']:
        for stmt in false_statements:
            honest = f"{user_tag} {template.format(type='an honest')} {assistant_tag} {stmt}".strip()
            dishonest = f"{user_tag} {template.format(type='a dishonest')} {assistant_tag} {stmt}".strip()
            combined.append((honest, dishonest, 0))  # 0 for false

    if shuffle:
        random.shuffle(combined)

    dataset = {'train': {'data': [], 'labels': [], 'honesty': []}, 'test': {'data': [], 'labels': [], 'honesty': []}}
    for i, pair in enumerate(combined):
        if i < n_train:
            key = 'train'
        else:
            key = 'test'
        dataset[key]['data'] += [pair[0], pair[1]]
        dataset[key]['labels'] += [pair[2], pair[2]]
        dataset[key]['honesty'] += [1, 0]

    print(f"Total pairs: {len(combined)}")
    print(f"Train samples: {len(dataset['train']['data'])}")
    print(f"Test samples: {len(dataset['test']['data'])}")
    return dataset


def happy_sad_dataset(data_path, shuffle=True, train_ratio=0.8, randomize_labels=False, seed=42):
    happiness_path = os.path.join(data_path, "happiness.json")
    sadness_path = os.path.join(data_path, "sadness.json")
    with open(happiness_path, 'r', encoding="utf-8") as f:
        happiness_data = json.load(f)
    with open(sadness_path, 'r', encoding="utf-8") as f:
        sadness_data = json.load(f)

    examples = []

    for sentence in happiness_data:
        assistant_response = sentence.strip() if isinstance(sentence, str) else sentence
        examples.append((assistant_response, 1))

    for sentence in sadness_data:
        assistant_response = sentence.strip() if isinstance(sentence, str) else sentence
        examples.append((assistant_response, 0))

    if shuffle:
        random.shuffle(examples)
    
    if randomize_labels:
        random.seed(seed)
        examples = [(ex[0], random.randint(0, 1)) for ex in examples] # replace with random labels

    return _split_examples(examples, train_ratio)


def load_commonsense(data_path, shuffle=False, train_ratio=0.75, n_sample=2000, randomize_labels=False, seed=42):
    assert shuffle == False, "Shuffling is not supported for commonsense dataset."
    df_train = pd.read_csv(data_path + '/cm_train.csv')
    df_test = pd.read_csv(data_path + '/cm_test.csv')

    df_all = pd.concat([df_train, df_test], ignore_index=True)
    df_all_short = df_all[df_all['is_short'] == True]
    df_all_short = _interleave_balanced(df_all_short)

    examples = [(row["input"], row["label"]) for _, row in df_all_short.iterrows()]

    if randomize_labels:
        random.seed(seed)
        examples = [(ex[0], random.randint(0, 1)) for ex in examples] # replace with random labels

    total_examples = len(examples)
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    effective_ratio = _compute_effective_ratio(n_sample, used_examples, train_ratio)
    return _split_examples(examples, effective_ratio)


def load_true_false(data_path, shuffle=True, train_ratio=0.75, n_sample=2000, randomize_labels=False, seed=42):
    df_all = pd.concat([
        pd.read_csv(data_path + '/' + f) for f in [
            'animals_true_false.csv', 'cities_true_false.csv', 'companies_true_false.csv',
            'elements_true_false.csv', 'facts_true_false.csv', 'generated_true_false.csv',
            'inventions_true_false.csv',
        ]
    ], ignore_index=True)

    if shuffle:
        df_all = df_all.sample(frac=1).reset_index(drop=True)
    df_all = _interleave_balanced(df_all)

    examples = [(row["statement"], row["label"]) for _, row in df_all.iterrows()]

    if randomize_labels:
        random.seed(seed)
        examples = [(ex[0], random.randint(0, 1)) for ex in examples] # replace with random labels

    total_examples = len(examples)
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    effective_ratio = _compute_effective_ratio(n_sample, used_examples, train_ratio)
    return _split_examples(examples, effective_ratio)



def load_simple_txt(data_path, shuffle=True, train_ratio=0.5, n_sample=1200):
    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            try:
                label = int(parts[-1])
            except ValueError:
                continue
            sentence = " ".join(parts[1:-1]).strip()
            if not sentence:
                continue
            entries.append((sentence, label))
    if not entries:
        raise ValueError(f"No usable rows found in {data_path}")
    if shuffle:
        random.shuffle(entries)

    examples = list(entries)
    total_examples = len(examples)
    if total_examples < 2:
        raise ValueError("Dataset requires at least two samples for train/test split.")
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    effective_ratio = _compute_effective_ratio(used_examples, used_examples, train_ratio, min_split=1)
    return _split_examples(examples, effective_ratio)


def load_sst2(data_path, shuffle=True, train_ratio=0.75, n_sample=2000, randomize_labels=False, seed=42):
    df = pd.read_csv(os.path.join(data_path, "sst2.csv"))
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    df = _interleave_balanced(df, label_col='label')

    examples = [(row["sentence"], row["label"]) for _, row in df.iterrows()]

    if randomize_labels:
        random.seed(seed)
        examples = [(ex[0], random.randint(0, 1)) for ex in examples]

    total_examples = len(examples)
    if n_sample is not None:
        examples = examples[:n_sample]
    used_examples = len(examples)

    if total_examples != used_examples:
        print(f"Total examples available: {total_examples}")
    print(f"Used examples: {used_examples}")

    effective_ratio = _compute_effective_ratio(n_sample, used_examples, train_ratio)
    return _split_examples(examples, effective_ratio)


def load_sycophancy_agree(data_path, shuffle=True, train_ratio=0.75, n_sample=1200):
    entries = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(' / ')]
            if len(parts) < 3:
                continue
            sentence = parts[1]
            label_part = next((part for part in parts if part.startswith('sycophancy=')), None)
            if label_part is None:
                continue
            try:
                label = int(label_part.split('=', 1)[1])
            except ValueError:
                continue
            entries.append((sentence, label))
    if not entries:
        raise ValueError(f"No usable rows found in {data_path}")
    if shuffle:
        random.shuffle(entries)
    if n_sample is None or n_sample <= 0:
        raise ValueError('n_sample must be a positive integer.')

    total_entries = len(entries)
    examples = entries[:min(n_sample, total_entries)]
    used_examples = len(examples)

    if used_examples < 2:
        raise ValueError('Sycophancy agreement dataset requires at least two samples for train/test split.')

    print(f'Total examples: {total_entries}')
    print(f'Used examples: {used_examples}')

    effective_ratio = _compute_effective_ratio(n_sample, used_examples, train_ratio, min_split=1)
    return _split_examples(examples, effective_ratio)


def emotion(data_path, shuffle=True):
    data = []
    all_json_files = glob(f"{data_path}/*.json")
    for file in all_json_files:
        with open(file, 'r') as f:
            data.extend(json.load(f))

    if shuffle:
        random.shuffle(data)
    return data


def load_dataset(dataset_name, n_sample=1500, n_test=600, randomize_labels=False, seed=42):
    seed_everything(42)  # for shuffle dataset; keep unchanged
    train_ratio = 1 - n_test / n_sample
    if dataset_name == "happy_sad":
        dataset = happy_sad_dataset("data/emotions", shuffle=True, train_ratio=train_ratio, randomize_labels=randomize_labels, seed=seed)
    elif dataset_name == "commonsense":
        dataset = load_commonsense('data/ethics_commonsense', shuffle=False, train_ratio=train_ratio, n_sample=n_sample, randomize_labels=randomize_labels, seed=seed)
        # do not shuffle commonsense dataset, making a balanced training set
    # elif dataset_name == "honesty":
    #     dataset = honesty_function_dataset("data/facts_true_false.csv", tags=tags, shuffle=False, n_train=512, include_tf='both')
    elif dataset_name == "true_false":
        dataset = load_true_false('data/true-false-dataset', shuffle=True, train_ratio=train_ratio, n_sample=n_sample, randomize_labels=randomize_labels, seed=seed)
    elif dataset_name == "power_seeking":
        dataset = load_simple_txt("data/power-seeking.txt", shuffle=True, train_ratio=train_ratio, n_sample=n_sample)
    # elif dataset_name == "sycophancy":
    #     dataset = load_simple_txt('data/sycophancy_dataset.txt', shuffle=True, train_ratio=0.5, n_sample=1200)
    elif dataset_name == "sycophancy":
        dataset = load_sycophancy_agree('data/sycophancy_agreement.txt', shuffle=True, train_ratio=train_ratio, n_sample=n_sample)
    elif dataset_name == "sst2":
        dataset = load_sst2('data/sst2', shuffle=True, train_ratio=train_ratio, n_sample=n_sample, randomize_labels=randomize_labels, seed=seed)
    else:
        raise ValueError(f"Unknown dataset {dataset_name}, please choose from happy_sad, commonsense, true_false, sycophancy, sst2.")

    return dataset


if __name__ == "__main__":
    # dt = happy_sad_dataset("../data/emotions", shuffle=True, train_ratio=0.6)
    dt = load_commonsense('../data/ethics_commonsense', shuffle=False, train_ratio=0.6, n_sample=1500)
    # dt = load_simple_txt("../data/power-seeking.txt", shuffle=True, train_ratio=0.6, n_sample=1500)
    # dt = load_true_false('../data/true-false-dataset', shuffle=True, train_ratio=0.6, n_sample=1500)
    # dt = load_sycophancy_agree('../data/sycophancy_agreement.txt', shuffle=True, train_ratio=0.6, n_sample=1500)
    print(dt["train"].keys())
    print(dt["train"]["sentences"][:5])
    print(dt["train"]["labels"][:5])
