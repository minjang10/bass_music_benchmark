import string
import re
from datetime import datetime
import json
import os
import numpy as np
import collections
import matplotlib.pyplot as plt
from json_repair import repair_json
import argparse

# Hungarian-algorithm-based IoU for comparing predicted vs ground truth segments
from utils.iou_metric import calculate_iou_hungarian as evaluate

def normalize_span_output(data):
    """
    Standardizes section names so that superficial differences (case, whitespace,
    punctuation, trailing numbers like "Verse 2") don't cause mismatches.
    Strips digits, punctuation (except hyphens), whitespace, and lowercases.
    """
    
    if not data:
        return data
    
    normalized = []
    for item in data:
        normalized_item = item.copy()
        
        if 'section' in normalized_item:
            section = normalized_item['section']
            section = re.sub(r'\d+', '', section)
            punctuation_to_remove = string.punctuation.replace('-', '')
            section = section.translate(str.maketrans('', '', punctuation_to_remove))
            section = section.lower()
            section = section.replace(' ', '').replace('\t', '').replace('\n', '')
            normalized_item['section'] = section
        
        normalized.append(normalized_item)
    
    return normalized

def extract_intervals_labels(int_labs, section=None):
    """
    Converts a list of section dicts (with 'start', 'end', and optionally 'section')
    into the (intervals, labels) format expected by mir_eval / the IoU metric.

    Handles timestamps in both numeric seconds and "MM:SS" string format.
    Falls back to float64 max on overflow (e.g., absurdly large predictions).

    Args:
        int_labs: List of dicts, each with 'start' and 'end' keys.
        section: If provided, all labels are set to this value (for single-section tasks).
                 Otherwise labels default to 'full'.

    Returns:
        (intervals, labels) — ndarray of shape (N,2) and list of N strings,
        or the string 'None' if parsing fails.
    """
    N = len(int_labs)
    intervals = np.ndarray((N, 2), dtype=np.float64)
    labels = []
    int_labs = normalize_span_output(int_labs)
    max_float = np.finfo(np.float64).max
    for i, section_obj in enumerate(int_labs):
        try:
            start = section_obj['start']
            end = section_obj['end']
            # Convert "MM:SS" timestamps to seconds
            if ":" in str(start):
                mins, secs = start.split(":", maxsplit=1)
                start = 60 * float(mins) + float(secs)
            if ":" in str(end):
                mins, secs = end.split(":", maxsplit=1)
                end = 60 * float(mins) + float(secs)
            intervals[i] = [start, end]
        except OverflowError:
            # Clamp to max float rather than crashing
            try:
                start = float(start)
            except OverflowError:
                start = max_float
            try:
                end = float(end)
            except OverflowError:
                end = max_float
            intervals[i] = [start, end]
        except (KeyError, ValueError, TypeError) as e:
            return str(None)
        if section:
            labels.append(section.lower().strip())
        else:
            labels.append('full')
    return intervals, labels

def validate_pred(int_labs, full=False):
    """
    Filters prediction dicts to only those with valid structure.
    Requires 'start' and 'end' keys; if full=True, also requires 'section'.
    Silently drops malformed entries.
    """
    valid_output = []
    for section in int_labs:
        try:
            if "start" not in section or "end" not in section:
                continue
            if full and "section" not in section:
                continue
            if not isinstance(section, dict):
                continue
            valid_output.append(section)
        except:
            pass
    return valid_output

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help="Path to the result file to evaluate")
    args = parser.parse_args()
    result_file = args.result_file
    dir_name = os.path.dirname(result_file).split('/')[-1]

    model_ious = {}
    model_ifrs = {}
    model_iou_std = {}
    # Per-section list of IoU scores, used for mean and std dev calculation
    iou_scores_per_task = collections.defaultdict(list)
    IFR_count = 0
    IFR_total = 0

    with open(result_file) as f:
        model_output = json.load(f)
        for question in model_output:
            task = question['task']
            # Two task modes:
            # - section_structural_segmentation: predict timestamps for ONE named section
            # - full structural segmentation: predict all sections with timestamps
            is_section_structural_segmentation = (task == 'section_structural_segmentation')

            if question['prediction'] is None:
                IFR_total += 1
                if is_section_structural_segmentation:
                    iou_scores_per_task[question['section'].lower()].append(0.0)
                else:
                    iou_scores_per_task['full'].append(0.0)
                continue
            
            # --- Parse ground truth into (intervals, labels) ---
            gt = question['gt']
            if is_section_structural_segmentation:
                ref = extract_intervals_labels(gt, question['section'].lower())
            else:
                ref = extract_intervals_labels(gt)
            ref_intervals, ref_labels = ref
            
            # --- Parse prediction: repair JSON, coerce to list of dicts ---
            pred = question['prediction']
            pred = repair_json(pred, return_objects=True)
            if isinstance(pred, dict):
                pred = [pred]
            if not isinstance(pred, list):
                continue

            # Validate and convert prediction to (intervals, labels)
            if is_section_structural_segmentation:
                curr_section = question['section'].lower()
                pred = validate_pred(pred, full=False)
                est = extract_intervals_labels(pred, curr_section)
                if est != str(None):
                    est_intervals, est_labels = est
                    IFR_count += 1
                    IFR_total += 1
                else:
                    IFR_total += 1
                    iou_scores_per_task[curr_section].append(0.0)
                    continue
            else:
                # Full segmentation requires 'section' key in each pred dict
                pred = validate_pred(pred, full=True)
                est = extract_intervals_labels(pred)
                if est != str(None):
                    est_intervals, est_labels = est
                    IFR_count += 1
                    IFR_total += 1
                else:
                    iou_scores_per_task['full'].append(0.0)
                    IFR_total += 1
                    continue

            # Compute IoU via Hungarian matching between ref and est segments
            iou = evaluate(ref_intervals, ref_labels, est_intervals, est_labels)[0]
            if question['task'] == 'section_structural_segmentation':
                curr_section = question['section'].lower()
                iou_scores_per_task[curr_section].append(iou)
            else:
                iou_scores_per_task['full'].append(iou)

        # Aggregate mean and std dev of IoU per section type
        iou_scores = {
            section: np.mean(iou_scores_per_task[section]) if iou_scores_per_task[section] else 0
            for section in iou_scores_per_task
        }
        iou_std_devs = {
            section: np.std(iou_scores_per_task[section]) if iou_scores_per_task[section] else 0
            for section in iou_scores_per_task
        }
        model_ious[dir_name] = iou_scores
        model_iou_std[dir_name] = iou_std_devs

        print(
            f"{dir_name} Structural Segmentation iou scores by Section: {iou_scores}"
        )
        ifr = (IFR_count/IFR_total) * 100 if IFR_total > 0 else 0
        print(f"{dir_name} IFR: {ifr:.2f}%")
        model_ifrs[dir_name] = ifr

    # Write summary: report full-song IoU and mean across individual sections
    print("\nWriting summary files...")
    section_ious = []
    accs = collections.defaultdict(dict)
    for task in model_ious[dir_name]:
        iou = model_ious[dir_name][task]
        std_dev = model_iou_std[dir_name][task]
        if task != 'full':
            section_ious.append(iou)
        if task == 'full':
            accs['full structural segmentation']['accuracy'] = iou
    if section_ious:
        accs['section structural segmentation']['accuracy'] = np.mean(section_ious)
    
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