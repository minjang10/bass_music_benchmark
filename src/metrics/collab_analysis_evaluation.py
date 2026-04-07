import re
import argparse
import ast
import collections
import json
import json_repair
import os
import matplotlib.pyplot as plt
import numpy as np
import gc
from decimal import Decimal

# Maps high-level evaluation categories to their constituent task types.
# Each category shares a common scoring strategy (e.g., exact match for counts,
# tolerance-based for durations, string match for collaboration types).
task_categories = {
    'artist_count': ['artist_count', 'artist_count_feature', 'timestamp_artist_count', 'section_artist_count', 'delivery_count'],
    'artist_duration': ['delivery_duration', 'artist_delivery_duration', 'artist_duration', 'section_duration'],
    'collab_type': ['section_function_identification', 'section_delivery_identification', 'timestamp_delivery_identification', 'delivery_comparison', 'delivery_comparison_hard'],
    'identify_artist_start_timestamp': ['identify_artist_start_timestamp']
}

def get_task_category(task):
    """
    Looks up which evaluation category a task belongs to.
    This determines which scoring logic (exact int match, ±3s tolerance,
    or string match) will be applied during evaluation.
    Returns None if the task isn't registered in any category.
    """
    for category, tasks in task_categories.items():
        if task in tasks:
            return category
    return None

def extract_last_number(text):
    """
    Pulls the last numeric value from free-form model output text.
    Handles both integers and floats. Uses the *last* match to catch cases
    where models produce reasoning text before a final answer.
    Returns int for whole numbers, float for decimals, or None if no number found.
    """
    if text is None:
        return None

    matches = re.findall(r'\d+(?:\.\d+)?', str(text))
    if not matches:
        return None

    num = matches[-1]
    return float(num) if '.' in num else int(num)

def convert_to_number(text):
    """
    Handles the case where a model spells out a digit as a word
    (e.g., "three" → 3). Only covers single digits 0–9.
    Returns the original text unchanged if it's not a recognized word.
    """
    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9
    }
    return number_words.get(text.lower(), text)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help="Name of the result file to evaluate")
    args = parser.parse_args()
    result_file = args.result_file
    dir_name = os.path.dirname(result_file).split('/')[-1]
    
    # Chance-level accuracy for each task, used to compute normalized scores.
    # Count and duration tasks have 0% random baseline (open-ended answers).
    # Classification tasks have 1/N baselines depending on number of choices.
    random_baseline = {
        'artist_count': {
            'artist_count': 0.0,
            'artist_count_feature': 0.0,
            'timestamp_artist_count': 0.0,
            'section_artist_count': 0.0,
            'delivery_count': 0.0
        },
        'artist_duration': {
            'artist_delivery_duration': 0.0,
            'artist_duration': 0.0,
            'delivery_duration': 0.0,
            'section_duration': 0.0
        },
        'collab_type': {
            'delivery_comparison': (1/2) * 100,
            'section_delivery_identification': (1/3) * 100,
            'section_function_identification': (1/3) * 100,
            'timestamp_delivery_identification': (1/3) * 100,
        },
        'identify_artist_start_timestamp': {
            'identify_artist_start_timestamp': 0.0,
        }
    }

    # Accumulators for per-model, per-category, per-task metrics
    model_accs = collections.defaultdict(dict)
    model_stds = collections.defaultdict(dict)
    model_accs_normalized = collections.defaultdict(dict)
    model_ifrs = {}

    # IFR (Instruction Following Rate): tracks how often the model produced
    # a parseable answer (as opposed to None or unparseable output).
    IFR_count = 0   # predictions that were parseable
    IFR_total = 0   # total questions attempted

    num_correct_per_task = collections.defaultdict(collections.Counter)
    total_per_task = collections.defaultdict(collections.Counter)
    # Stores per-question binary results (1.0/0.0) for std deviation calculation
    all_results_per_task = collections.defaultdict(lambda: collections.defaultdict(list))

    with open(result_file) as f:
        model_output = json.load(f)
        for question in model_output:
            task = question['task']
            cat = get_task_category(task)

            # Null prediction: model failed to produce any output
            if question['prediction'] is None:
                total_per_task[cat][task] += 1
                all_results_per_task[cat][task].append(0.0)
                IFR_total += 1
                continue

            gt = question['gt']
            correct = False

            if cat == 'artist_count':
                # --- Count tasks: exact integer match required ---
                prediction = convert_to_number(question['prediction'])
                prediction = extract_last_number(prediction)
                if prediction is None:
                    total_per_task[cat][task] += 1
                    all_results_per_task[cat][task].append(0.0)
                    IFR_total += 1
                    continue
                prediction = int(prediction)
                correct = (gt == prediction)

            elif cat == 'artist_duration' or task == 'identify_artist_start_timestamp':
                # --- Temporal tasks: correct if within ±3 seconds of ground truth ---
                prediction = convert_to_number(question['prediction'])
                prediction = extract_last_number(prediction)
                if prediction is None:
                    total_per_task[cat][task] += 1
                    all_results_per_task[cat][task].append(0.0)
                    IFR_total += 1
                    continue
                try:
                    prediction = float(prediction)
                    correct = (abs(gt - prediction) <= 3)
                except OverflowError:
                    # Fallback to Decimal for extremely large predicted values
                    prediction = Decimal(str(prediction))
                    gt_decimal = Decimal(str(gt))
                    correct = (abs(gt_decimal - prediction) <= Decimal('3'))

            elif cat == 'collab_type':
                # --- Classification tasks: case-insensitive exact string match ---
                # Strip everything except letters and spaces before comparing
                prediction = question['prediction'].lower().strip()
                prediction = ''.join(char for char in prediction if char.isalpha() or char.isspace())
                gt_processed = ''.join(char for char in gt.lower().strip() if char.isalpha() or char.isspace())
                correct = (gt_processed == prediction)
            else:
                print("Unknown task category:", task)
                exit()
            
            if correct:
                num_correct_per_task[cat][task] += 1
            total_per_task[cat][task] += 1
            all_results_per_task[cat][task].append(1.0 if correct else 0.0)
            IFR_count += 1
            IFR_total += 1
                    
        # IFR: percentage of questions where the model gave a parseable answer
        IFR = (IFR_count / IFR_total) * 100 if IFR_total > 0 else 0
        model_ifrs[dir_name] = IFR
        print(f'{dir_name} IFR: {IFR:.2f}%')

        # Compute per-task accuracy, normalized accuracy (adjusted for random
        # baseline so that chance = 0% and perfect = 100%), and std deviation
        for cat in total_per_task:
            task_accs = {task: (num_correct_per_task[cat][task] / total_per_task[cat][task]) * 100 for task in total_per_task[cat]}
            task_accs_normalized = {task: (acc - random_baseline[cat][task]) / (100 - random_baseline[cat][task]) * 100 for task, acc in task_accs.items()}
            task_stds = {task: np.std(all_results_per_task[cat][task]) * 100 for task in all_results_per_task[cat]}
            tasks_stds_normalized = {task: ((task_stds[task] - random_baseline[cat][task]) / (100 - random_baseline[cat][task])) * 100 for task in task_stds}
            model_accs[cat][dir_name] = task_accs
            model_stds[cat][dir_name] = task_stds
            model_accs_normalized[cat][dir_name] = task_accs_normalized
            
            acc_values = list(task_accs.values())
            mean_acc = np.mean(acc_values)
            std_acc = np.std(acc_values)
            print(f'{dir_name} - {cat}: Mean={mean_acc:.2f}%, StdDev={std_acc:.2f}%')
        
    # Write a summary JSON with mean accuracy and normalized accuracy per category
    print("Writing summary file...")
    accs = collections.defaultdict(dict)
    for cat in model_accs:
        category_accuracies = []
        category_accuracies_normalized = []
        accuracy = model_accs[cat][dir_name]
        stddev = model_stds[cat][dir_name]
        normalized_accuracy = model_accs_normalized[cat][dir_name]
        for task in accuracy:
            category_accuracies.append(accuracy[task])
            category_accuracies_normalized.append(normalized_accuracy[task])
    
        mean_accuracy = np.mean(category_accuracies)
        mean_normalized_accuracy = np.mean(category_accuracies_normalized)
        accs[cat]["accuracy"] = mean_accuracy
        accs[cat]["normalized_accuracy"] = mean_normalized_accuracy

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
