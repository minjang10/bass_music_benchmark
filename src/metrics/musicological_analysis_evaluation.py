import json
import ast
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
import argparse
import re

def extract_last_number(text):
    """
    Extracts the last integer or float from a string, including negatives.
    Uses the *last* match to handle models that produce reasoning before a final answer.
    Returns int for whole numbers, float for decimals, or None if nothing found.
    """
    if text is None:
        return None

    matches = re.findall(r'-?\d+(?:\.\d+)?', str(text))
    if not matches:
        return None

    num = matches[-1]
    return float(num) if '.' in num else int(num)

def convert_to_number(text):
    """
    Converts a spelled-out single digit ("three" → 3). Only covers 0–9.
    Returns the original text unchanged if it's not a recognized word.
    """
    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    return number_words.get(str(text).lower(), text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help="Path to the result file to evaluate")
    args = parser.parse_args()
    result_file = args.result_file
    dir_name = os.path.dirname(result_file).split('/')[-1]
    
    # Chance-level accuracy for each task type, used to compute normalized scores.
    # ranking_genes: 1/4! = 1/24 (permutation of 4 items)
    # pair selection: 1/C(12,2) = 1/66 (choosing 2 from 12 options)
    # MCQ tasks: 1/4 (four answer choices)
    random_baselines = {
        'ranking_genes': (1 / 24),
        'pair selection': (1 / 66),
        'multi_song_mcq_max': (1 / 4),
        'multiple choice question': (1 / 4),
        'multiple choice question (easy)': (1 / 4)
    }

    model_acc = {}
    model_std = {}
    model_ifrs = {}
    pairing_per_gene_correct = {}
    pairing_per_gene_total = {}
    total = 0

    # IFR (Instruction Following Rate): fraction of questions where the model
    # produced a parseable, valid answer
    IFR_count = 0
    IFR_total = 0

    num_correct_per_task = collections.Counter()
    total_per_task = collections.Counter()
    all_results_per_task = collections.defaultdict(list)  # binary results for std dev

    with open(result_file) as f:
        model_output = json.load(f)
        for question in model_output:
            # Null prediction: model failed entirely
            if question['prediction'] is None:
                IFR_total += 1
                total_per_task[question['task']] += 1
                all_results_per_task[question['task']].append(0.0)
                continue
            
            # Parse ground truth — stored as string for some tasks
            gt = question['gt']
            if isinstance(gt, str) and 'multiple choice question' not in question['task']:
                gt = ast.literal_eval(gt)

            pred = question['prediction']

            # Handle ensemble predictions (list of 4 attempts → majority vote)
            if question['task'] != 'multi_song_mcq_max':
                if isinstance(pred, list) and len(pred) == 4:
                    pred = str(max(set(pred), key=pred.count))
            else:
                # multi_song_mcq_max: unwrap single-element list
                if isinstance(pred, list) and len(pred) == 1:
                    pred = str(pred[0])
                    
            # Try to parse string predictions as Python literals (handles JSON-like output)
            if isinstance(pred, str):
                try:
                    pred = ast.literal_eval(pred.removeprefix("```json").removesuffix("```").strip())
                except (SyntaxError, ValueError):
                    pass
            if pred is None:
                total_per_task[question['task']] += 1
                all_results_per_task[question['task']].append(0.0)
                IFR_total += 1
                continue
                    
            
            correct = False

            if "multiple choice question" in question['task'].lower():
                # --- MCQ: check if ground truth answer appears in prediction ---
                # Handles both numeric (1-4 index) and text predictions
                if isinstance(pred, int) or isinstance(pred, float):
                    pred = int(pred)
                    if 1 <= pred <= 4:
                        # Convert 1-indexed choice number to the actual answer text
                        pred = question['answer_choices'][pred - 1]
                    else:
                        IFR_total += 1
                        all_results_per_task[question['task']].append(0.0)
                        total_per_task[question['task']] += 1
                        continue
                if isinstance(pred, str):
                    pred = pred.lower().strip()
                # Substring match: correct if gt appears anywhere in prediction
                correct = (gt.lower().strip() in pred)

            elif question['task'] == 'pair selection':
                # --- Pair selection: model must identify exactly 2 matching items ---
                if isinstance(pred, list) and len(pred) == 2:
                    # List output: order-independent set comparison
                    pred_lower = [p.lower().strip() for p in pred]
                    gt_lower = [g.lower().strip() for g in gt]
                    correct = sorted(pred_lower) == sorted(gt_lower)
                elif isinstance(pred, str):
                    # String output: both correct items must appear AND no distractors
                    pred = pred.lower().strip()
                    distractors = [a.lower().strip() for a in question['answer_choices']]
                    distractors.remove(gt[0].lower().strip())
                    distractors.remove(gt[1].lower().strip())
                    correct = all(g.lower().strip() in pred for g in gt) and all(a.lower().strip() not in pred for a in distractors)

            elif question['task'] == 'multi_song_mcq_max':
                # --- Multi-song MCQ: expects a 1-indexed integer answer ---
                if not isinstance(pred, int):
                    pred = convert_to_number(pred)
                    pred = extract_last_number(pred)
                if not pred:
                    IFR_total += 1
                    total_per_task[question['task']] += 1
                    all_results_per_task[question['task']].append(0.0)
                    continue
                pred = int(pred)
                if 1 <= pred <= 4:
                    correct = (gt == pred)

            elif question['task'] == 'ranking_genes':
                # --- Gene ranking: model must produce all 4 items in exact order ---
                if isinstance(pred, list) and len(pred) == 4:
                    pred = [p.lower().strip() for p in pred if isinstance(p, str)]
                    gt_lower = [g.lower().strip() for g in gt]
                    correct = (pred == gt_lower)
                elif isinstance(pred, str):
                    # String fallback: verify all items appear in correct sequential order
                    pred = pred.lower().strip()
                    gt_lower = [g.lower().strip() for g in gt]
                    current_index = -1
                    correct = True
                    for g in gt_lower:
                        index = pred.find(g)
                        if index == -1 or index < current_index:
                            correct = False
                            break
                        current_index = index
                        
            task = question['task']
            # Collapse "multiple choice question" and "multiple choice question (easy)"
            # into a single task for reporting
            if 'multiple choice question' in task:
                task = 'multiple choice question'
            if correct:
                num_correct_per_task[task] += 1
            total_per_task[task] += 1
            all_results_per_task[task].append(1.0 if correct else 0.0)
            IFR_count += 1
            IFR_total += 1

        IFR = (IFR_count / IFR_total) * 100 if IFR_total > 0 else 0
        print(f"{dir_name} IFR: {IFR:.2f}%")
        model_ifrs[dir_name] = IFR

        # Raw accuracy per task (0–1 scale, not percentage)
        acc_per_task = {
            task: (num_correct_per_task[task] / total_per_task[task]) if total_per_task[task] > 0 else 0
            for task in total_per_task
        }
        
        std_per_task = {
            task: np.std(all_results_per_task[task])
            for task in all_results_per_task
        }
        
        acc_values = list(acc_per_task.values())
        if acc_values:
            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            print(f'{dir_name} - Overall: Mean={mean_acc:.3f}, StdDev={std_acc:.3f}')

        print(f'{dir_name} Accuracy per task:\n{acc_per_task}')
        model_acc[dir_name] = acc_per_task

        # Normalized accuracy: rescales so that random baseline = 0, perfect = 1
        model_acc[dir_name + '-normalized'] = {
            task: ((acc_per_task[task] - random_baselines[task]) / (1 - random_baselines[task]))
            for task in acc_per_task
        }
        model_std[dir_name + '-normalized'] = {
            task: ((std_per_task[task] - random_baselines.get(task, 0)) / (1 - random_baselines.get(task, 0))) if (1 - random_baselines.get(task, 0)) != 0 else 0
            for task in std_per_task
        }
        model_std[dir_name] = std_per_task
    
    # Build final summary dict
    accs = collections.defaultdict(dict)
    if model_acc[dir_name]:
        mean_acc = model_acc[dir_name]
        std_acc = model_std[dir_name]
        mean_acc_normalized = model_acc[dir_name + '-normalized']
        print(os.path.basename(dir_name), "MEAN ACC:", mean_acc)
        print(os.path.basename(dir_name), "NORMALIZED ACC", mean_acc_normalized)
        for task in mean_acc:
            accs[task]["accuracy"] = mean_acc[task]
            accs[task]["normalized_accuracy"] = mean_acc_normalized[task]

    if os.path.exists(f"{dir_name}_summary.json"):
        with open(f"{dir_name}_summary.json", "r") as f:
            existing_data = json.load(f)
    else:
        existing_data = []
    existing_data.append(accs)
    with open(f"{dir_name}_summary.json", "w") as f:
        json.dump(existing_data, f, indent=4)
if __name__ == '__main__':
    main()