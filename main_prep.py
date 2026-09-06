"""Prepare the scores for later experiments."""

import os
from pathlib import Path
import pandas as pd
import torch
from joblib import load
import argparse
from analysis.process_hidden import get_hiddens, train_classify_hiddens, eval_classify_hiddens
from data.load import load_dataset
from utils import seed_everything, load_lm, load_exp_cfg, safe_dump, load_yaml
from plotter import plot_neural_classifier_accuracies, plot_neuro_scores_distribution

ROOT = Path(__file__).resolve().parents[0]


def apply_chat_template_to_dataset(dataset, tokenizer):
    prompt = load_yaml(ROOT / "configs" / "prompts.yml")

    def format_chat_prompt(assistant_response):
        messages = [
            {"role": "system", "content": prompt["system"]},
            {"role": "user", "content": prompt["user_msg"]},
            {"role": "assistant", "content": assistant_response},
        ]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    for split in ("train", "test"):
        split_data = dataset[split]
        assistant_data = split_data["sentences"]
        split_data["full_prompt"] = [format_chat_prompt(assistant) for assistant in assistant_data]
    return dataset


def save_ICL_inputs(model, tokenizer, dataset, cfg, save_dir, randomize=False):
    data_dict = {"train": {}, "test": {}}
    for partition in ['train', 'test']:
        assert len(dataset[partition]['full_prompt']) > 0,'No data found under full_prompt key.'
        logits, hiddens = get_hiddens(model, tokenizer, dataset[partition]['full_prompt'],
                                      cfg.batch_size, cfg.process_hidden_method)
        data_dict[partition]["X"] = hiddens
        data_dict[partition]["y"] = dataset[partition]['labels']
    if not randomize:
        safe_dump(data_dict, save_dir / f"hidden_{cfg.process_hidden_method}_data_Xy.pkl")
    else:
        safe_dump(data_dict, save_dir / f"hidden_{cfg.process_hidden_method}_data_Xy_randomized.pkl")
    print("Data prepared and saved.")


def train_classifier(cfg, save_dir, file_name, eval_lr_clf=False, randomize=False):
    if not randomize:
        data_dict = load(save_dir / f"hidden_{cfg.process_hidden_method}_data_Xy.pkl")
    else:
        data_dict = load(save_dir / f"hidden_{cfg.process_hidden_method}_data_Xy_randomized.pkl")
        file_name += "_randomized"
    print("filename", file_name)
    all_classifiers, all_train_accuracies = train_classify_hiddens(
        data_dict['train']["X"], data_dict['train']["y"], cfg.clf, cfg.normalize, cfg.pc_number)

    safe_dump(all_classifiers, save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{file_name}.pkl")
    print("Classifiers trained and saved.")

    if eval_lr_clf:
        train_X, train_y = data_dict["train"]["X"], data_dict["train"]["y"]
        test_X, test_y = data_dict["test"]["X"], data_dict["test"]["y"]
        all_train_accuracies = eval_classify_hiddens(train_X, all_classifiers, return_type='accuracy',
                                                     labels=train_y, pc_number=cfg.pc_number)
        all_test_accuracies = eval_classify_hiddens(test_X, all_classifiers, return_type='accuracy',
                                                    labels=test_y, pc_number=cfg.pc_number)

        plot_neural_classifier_accuracies(list(all_train_accuracies.keys()), list(all_train_accuracies.values()),
                                         list(all_test_accuracies.values()), file_name, cfg.process_hidden_method, save_dir)

        all_test_scores = eval_classify_hiddens(test_X, all_classifiers, return_type='score',
                                                labels=test_y, pc_number=cfg.pc_number)
        save_file = save_dir / f"hidden_{cfg.process_hidden_method}_test_scores_{file_name}"
        plot_neuro_scores_distribution(list(all_classifiers.keys()), all_test_scores, save_file)


def generate_ICL_example_scores(dataset, model, tokenizer, cfg, save_dir, file_name, seed=42, randomize=False):
    """Generate baseline (one sentence) neurofeedback scores (given the axis) for all examples in ICL exp."""
    seed_everything(seed)
    if randomize:
        file_name += "_randomized"
    if 'pcascore' in file_name:
        # since all pcascore classifiers are the same, only train once
        all_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_pcascore_pc1.pkl")
    else:
        all_classifiers = load(save_dir / f"hidden_{cfg.process_hidden_method}_classifiers_{file_name}.pkl")
    full_prompts = dataset['test']['full_prompt']
    logits, hiddens = get_hiddens(model, tokenizer, full_prompts, cfg.batch_size, cfg.process_hidden_method)
    scores = eval_classify_hiddens(hiddens, all_classifiers, return_type='score', pc_number=cfg.pc_number)  # scores[layer][seq_idx]
    scores = pd.DataFrame(scores)
    scores['sentences'] = dataset['test']['sentences']
    save_file = save_dir / f"hidden_{cfg.process_hidden_method}_{file_name}_example_scores"
    plot_neuro_scores_distribution(list(all_classifiers.keys()), scores, save_file)
    safe_dump(scores, f'{save_file}.pkl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="prepare scores")
    parser.add_argument("--model", type=str, default="llama3_3b")
    parser.add_argument("--dataset", type=str, default="commonsense")  # commonsense, true_false, sycophancy
    parser.add_argument("--randomize", action='store_true', help="whether to randomize the labels as a control")
    parser.add_argument("--n_sample", type=int, default=1500, help="total number of samples to load") # 7000
    parser.add_argument("--n_test", type=int, default=600, help="number of test samples (remainder goes to train)") # 6000

    # python main_prep.py --model llama3.1_8b --dataset commonsense
    args = parser.parse_args()
    cfg = load_exp_cfg(args.model)
    save_dir = Path("results") / cfg.model_name.replace("/", "_") / args.dataset
    os.makedirs(save_dir, exist_ok=True)

    model, tokenizer = load_lm(cfg.model_name)
    dataset = load_dataset(args.dataset, n_sample=args.n_sample, n_test=args.n_test, randomize_labels=args.randomize)
    dataset = apply_chat_template_to_dataset(dataset, tokenizer)
    save_ICL_inputs(model, tokenizer, dataset, cfg, save_dir, randomize=args.randomize)

    cfg.clf = file_name = "lr"
    cfg.pc_number = None
    train_classifier(cfg, save_dir, file_name, True, randomize=args.randomize)
    generate_ICL_example_scores(dataset, model, tokenizer, cfg, save_dir, file_name, randomize=args.randomize)

    if not args.randomize:
        cfg.clf = "pcascore"
        cfg.pc_number = 1
        file_name = f'{cfg.clf}_pc{cfg.pc_number}'
        train_classifier(cfg, save_dir, file_name)  # only need to train once for all pcs
        for pc_number in cfg.all_pc_exp:
            cfg.pc_number = pc_number
            file_name = f'{cfg.clf}_pc{cfg.pc_number}'
            generate_ICL_example_scores(dataset, model, tokenizer, cfg, save_dir, file_name)
