import ast
import json
import os
import numpy as np
import jiwer
from jiwer import wer, cer
import collections
import matplotlib
import matplotlib.pyplot as plt
import re
import json_repair
from json_repair import repair_json
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple
import threading

def get_lyrics(lyrics):
    """
    Extracts just the lyrics strings from a list of section dictionaries,
    preserving their original order.
    Note: contains a bug — `result` is initialized as a dict but used as a list.
    """
    result = {}
    for row in lyrics:
        result.append(row['lyrics'])
        
    return result

def deep_clean(text):
    """
    Aggressively normalizes a section name: lowercases, strips ordinal prefixes
    ("1st ", "2nd ", "3rd "), and removes everything except lowercase letters
    and hyphens. Used to match section names across ground truth and predictions.
    """
    return re.sub(r"[^a-z-]", "", text.lower().replace("1st ", "").replace("2nd ", "").replace("3rd ", ""))

def clean(text):
    """
    Light normalization: lowercases and keeps only alphanumeric chars, hyphens,
    and spaces. Less aggressive than deep_clean — preserves digits and whitespace.
    """
    return re.sub(r"[^a-z0-9-\s]", "", text.lower())

def number_sections(pred):
    """
    Disambiguates repeated section names by appending an occurrence index.
    E.g., if "chorus" appears 3 times, they become "chorus1", "chorus2", "chorus3".
    Sections that only appear once are left unchanged.
    Returns 'None' (as string) if any section dict is missing a 'section' key.
    """
    total_section_counts = collections.Counter()
    for section_trans in pred:
        if 'section' not in section_trans:
            return str(None)

        total_section_counts[deep_clean(section_trans['section'])] += 1

    curr_count = collections.Counter()
    for section_trans in pred:
        if total_section_counts[deep_clean(section_trans['section'])] > 1:
            curr_count[deep_clean(section_trans['section'])] += 1
            section_trans['section'] = deep_clean(section_trans['section']) + str(curr_count[deep_clean(section_trans['section'])])

def get_wer(gt, pred, full):
    """
    Computes Word Error Rate between ground truth and prediction.

    Two modes controlled by `full`:
    - full=False (section-level): zips gt and pred lists pairwise by position.
    - full=True (full transcription): matches sections by cleaned name, so
      reordered or extra/missing sections are handled gracefully.

    Returns two WER values:
    - strict: missing/extra sections count as errors (empty string compared).
    - lenient: only scores sections present in both gt and pred.

    Special case: if gt and pred are both plain strings (no-section task),
    computes a single WER after normalizing whitespace/punctuation.
    """
    if type(gt[0]) == str and type(pred) == str:
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemovePunctuation(),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
        ])
        gt_transcript = transformation(gt[0].lower().replace('\n', ' '))
        pred_transcript = transformation(pred.lower().replace('\n', ' '))
        wer_value = wer(gt_transcript, pred_transcript)
        return wer_value
        
    wers_strict = []
    wers_lenient = []
    if full:
        # Build lookup dicts keyed by cleaned section name
        ref_sections = collections.defaultdict(str)
        pred_sections = collections.defaultdict(str)
        success = number_sections(pred)
        if success == 'None':
            return str(None)

        for d in gt:
            ref_sections[clean(d['section'])] = d['lyrics'].lower()
        for d in pred:
            if 'section' not in d or 'lyrics' not in d:
                return str(None)
            if d['lyrics'] is None: d['lyrics'] = ''
            if isinstance(d['lyrics'], list):
                if not isinstance(d['lyrics'][0], str):
                    return str(None)
                d['lyrics'] = ' '.join(d['lyrics'])
            pred_sections[clean(d['section'])] = d['lyrics'].lower()

        # Union of all section names from both sides
        all_sections = set(ref_sections) | set(pred_sections)
        for section in all_sections:
            # Strict: missing sections get compared against empty string → high WER
            ref = ref_sections.get(section, '')
            pred = pred_sections.get(section, '')
            wers_strict.append(wer(ref.lower().replace('\n', ' '), pred.lower().replace('\n', ' ')))
            
            # Lenient: skip sections that don't exist on both sides
            ref = ref_sections.get(section, None)
            pred = pred_sections.get(section, None)
            if not ref or not pred:
                continue
            wers_lenient.append(wer(ref.lower().replace('\n', ' '), pred.lower().replace('\n', ' ')))
    else:
        # Non-full mode: compare pairwise by position (gt[i] vs pred[i])
        for r, p in zip(gt, pred):
            if 'lyrics' not in p:
                return str(None)
            if isinstance(p['lyrics'], list):
                if not isinstance(p['lyrics'][0], str):
                    return str(None)
                p['lyrics'] = ' '.join(p['lyrics'])
            wers_strict.append(wer(r['lyrics'].lower().replace('\n', ' '), p['lyrics'].lower().replace('\n', ' ')))
            
            if not r or not p:
                continue
            wers_lenient.append(wer(r['lyrics'].lower().replace('\n', ' '), p['lyrics'].lower().replace('\n', ' ')))

    avg_wer_strict = sum(wers_strict) / len(wers_strict) if wers_strict else 0.0
    avg_wer_lenient = sum(wers_lenient) / len(wers_lenient) if wers_lenient else 0.0
    return avg_wer_strict, avg_wer_lenient


def process_question_chunk(questions: List[Dict]) -> Tuple[Dict, Dict, Dict, Dict, int, int]:
    """
    Scores a batch of questions, accumulating WER by section name.

    Handles three question formats:
    1. Plain text transcription (no sections) — scored as simple WER under "ASR".
    2. Section-level transcription — each question targets one named section.
    3. Full transcription — the model must produce all sections; matched by name.

    Prediction parsing is lenient: tries json.loads, falls back to json_repair,
    and coerces malformed outputs into the expected [{section, lyrics}] format.

    Returns per-section WER sums, per-section WER lists (for std dev),
    section counts, and IFR (instruction-following rate) counters.
    """
    wer_by_section_strict = collections.Counter()
    all_wers_by_section_strict = collections.defaultdict(list)
    wer_by_section_lenient = collections.Counter()
    all_wers_by_section_lenient = collections.defaultdict(list)
    section_counts = collections.Counter()
    IFR_count = 0
    IFR_total = 0

    for question in questions:
        # Null prediction means model produced no output at all
        if question['prediction'] is None:
            IFR_total += 1
            continue
        
        # --- Plain text transcription (no section structure expected) ---
        if question['task'] == 'no section lyrical transcription':
            asr_wer = get_wer(question['gt'], question['prediction'], False)
            if asr_wer is None:
                IFR_total += 1
                continue
            wer_by_section_strict['ASR'] += asr_wer
            all_wers_by_section_strict['ASR'].append(asr_wer)
            section_counts['ASR'] += 1
            wer_by_section_lenient['ASR'] += asr_wer
            all_wers_by_section_lenient['ASR'].append(asr_wer)
            
            IFR_count += 1
            IFR_total += 1
            continue
            
        # --- Structured transcription (section-level or full) ---
        gt = question['gt']
        if type(gt) == dict:
            gt = [gt]

        # Try to parse prediction as JSON; fall back to json_repair for malformed output
        pred = question['prediction'].removeprefix('```json').removesuffix('```').strip()
        try:
            pred = json.loads(pred)
        except json.JSONDecodeError:
            pred = json_repair.loads(pred)
        if not pred:
            # Last resort: strip markdown header and treat as raw text
            pred = question['prediction']
            pred.removeprefix("## Transcription:")
        if isinstance(pred, dict):
            pred = [pred]
        if not isinstance(pred, list):
            # Model gave unstructured text; wrap it so scoring can proceed
            pred = [
                {"section": "", "lyrics": pred}
            ]
        
        # Normalize each section dict to guarantee 'section' and 'lyrics' keys
        predictions = []
        for section_trans in pred:
            if not isinstance(section_trans, dict):
                section_trans = {"section": "", "lyrics": str(section_trans)}
            if "section" not in section_trans:
                section_trans["section"] = ""
            if "lyrics" not in section_trans:
                section_trans["lyrics"] = ""
            predictions.append(section_trans)
        pred = predictions
        if len(pred) == 0:
            continue
            
        is_full_transcription = 'full' in question['task']
        wer_result = get_wer(gt, pred, is_full_transcription)
        if wer_result == 'None':
            IFR_total += 1
            continue
        wer_strict, wer_lenient = wer_result
        IFR_count += 1
        IFR_total += 1

        # Accumulate WER under the appropriate section key
        if not is_full_transcription:
            # Task name format: "<section_name> section transcription" — extract section name
            curr_section = question['task'].rsplit(" ", maxsplit=2)[0]
            wer_by_section_strict[curr_section] += wer_strict
            all_wers_by_section_strict[curr_section].append(wer_strict)
            all_wers_by_section_lenient[curr_section].append(wer_lenient)
            section_counts[curr_section] += 1
        else:
            wer_by_section_strict['full'] += wer_strict
            all_wers_by_section_strict['full'].append(wer_strict)
            wer_by_section_lenient['full'] += wer_lenient
            all_wers_by_section_lenient['full'].append(wer_lenient)
            section_counts['full'] += 1
        with open("dev.txt", "a") as f:
            f.write("Processed a question in chunk\n")
        
    
    return (wer_by_section_strict, all_wers_by_section_strict, 
            wer_by_section_lenient, all_wers_by_section_lenient,
            section_counts, IFR_count, IFR_total)


def merge_results(results_list: List[Tuple]) -> Tuple:
    """
    Combines results from parallel worker chunks into a single set of
    aggregated counters and lists. Sums WER totals, extends per-section
    WER lists, and totals IFR counters.
    """
    merged_wer_strict = collections.Counter()
    merged_all_wers_strict = collections.defaultdict(list)
    merged_wer_lenient = collections.Counter()
    merged_all_wers_lenient = collections.defaultdict(list)
    merged_section_counts = collections.Counter()
    total_IFR_count = 0
    total_IFR_total = 0
    
    for (wer_strict, all_wers_strict, wer_lenient, all_wers_lenient, 
         section_counts, ifr_count, ifr_total) in results_list:
        merged_wer_strict.update(wer_strict)
        merged_wer_lenient.update(wer_lenient)
        merged_section_counts.update(section_counts)
        
        for section, wers in all_wers_strict.items():
            merged_all_wers_strict[section].extend(wers)
        for section, wers in all_wers_lenient.items():
            merged_all_wers_lenient[section].extend(wers)
        
        total_IFR_count += ifr_count
        total_IFR_total += ifr_total
    
    return (merged_wer_strict, merged_all_wers_strict,
            merged_wer_lenient, merged_all_wers_lenient,
            merged_section_counts, total_IFR_count, total_IFR_total)


def process_file_parallel(file_path: str, num_workers: int = None) -> Tuple:
    """
    Loads a results JSON and distributes its questions across worker threads.
    Uses ThreadPoolExecutor (not processes) since the work is largely I/O-bound
    (JSON parsing, file writes). Defaults to 4× CPU count workers.
    """
    with open(file_path) as f:
        model_output = json.load(f)
    
    with open("dev.txt", "a") as f:
        f.write(f"CPU Cores Count for {file_path}: {os.cpu_count()}\n")
    if num_workers is None:
        num_workers = min((os.cpu_count() or 1) * 4, len(model_output))
    
    with open("dev.txt", "a") as f:
        f.write(f"  Using {num_workers} workers for {len(model_output)} questions\n")
    
    # Split into roughly equal chunks, one per worker
    chunk_size = max(1, len(model_output) // num_workers)
    chunks = [model_output[i:i + chunk_size] 
              for i in range(0, len(model_output), chunk_size)]
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_question_chunk, chunk) 
                   for chunk in chunks]
        
        results = []
        for future in as_completed(futures):
            results.append(future.result())
    
    return merge_results(results)

        
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--result-file', type=str, required=True, help="Name of the result file to evaluate")
    args = parser.parse_args()
    result_file = args.result_file
    dir_name = os.path.dirname(result_file).split('/')[-1]

    model_wers = {}
    model_ifrs = {}
    model_std_devs = {}
    
    with open("dev.txt", "a") as f:
        f.write(f"Starting processing for file: {result_file}\n")
    
    # Accumulators that would support multi-file aggregation (currently single-file)
    all_all_wers_by_section_strict = collections.defaultdict(list)
    all_wer_by_section_strict = collections.Counter()
    all_wer_by_section_lenient = collections.Counter()
    all_all_wers_by_section_lenient = collections.defaultdict(list)
    all_section_counts = collections.Counter()
    all_IFR_count = 0
    all_IFR_total = 0
            
    (wer_strict, all_wers_strict, wer_lenient, all_wers_lenient,
        section_counts, ifr_count, ifr_total) = process_file_parallel(file_path=result_file)
    
    all_wer_by_section_strict.update(wer_strict)
    all_wer_by_section_lenient.update(wer_lenient)
    all_section_counts.update(section_counts)
    
    for section, wers in all_wers_strict.items():
        all_all_wers_by_section_strict[section].extend(wers)
    for section, wers in all_wers_lenient.items():
        all_all_wers_by_section_lenient[section].extend(wers)
    
    all_IFR_count += ifr_count
    all_IFR_total += ifr_total
        
    # Compute mean and std dev of WER per section, for both strict and lenient
    avg_wer_by_section_strict = {
        section: all_wer_by_section_strict[section] / all_section_counts[section]
        for section in all_wer_by_section_strict
    }
    print("ALL WERS:", all_all_wers_by_section_strict)
    std_dev_wer_by_section_strict = {
        section: np.std(all_all_wers_by_section_strict[section])
        for section in all_all_wers_by_section_strict
    }
    avg_wer_by_section_lenient = {
        section: all_wer_by_section_lenient[section] / all_section_counts[section]
        for section in all_wer_by_section_lenient
    }
    std_dev_wer_by_section_lenient = {
        section: np.std(all_all_wers_by_section_lenient[section])
        for section in all_all_wers_by_section_lenient
    }
    
    model_wers[dir_name + '-strict'] = avg_wer_by_section_strict
    model_std_devs[dir_name + '-strict'] = std_dev_wer_by_section_strict
    model_wers[dir_name + '-lenient'] = avg_wer_by_section_lenient
    model_std_devs[dir_name + '-lenient'] = std_dev_wer_by_section_lenient

    ifr = (all_IFR_count/all_IFR_total) * 100 if all_IFR_total > 0 else 0
    model_ifrs[dir_name] = ifr

    # Build summary: report WER and IWER (inverted WER = 1/(1+WER), so higher = better)
    # for full transcription and averaged across individual sections
    print("\nWriting summary files...")
    section_wers = []
    section_normalized_wers = []
    accs = collections.defaultdict(dict)
    for section in model_wers[dir_name + '-strict']:
        model_wer = model_wers[dir_name + '-strict'][section]
        model_std = model_std_devs[dir_name + '-strict'][section]
        # IWER: inverted WER, maps [0,∞) → (0,1] so higher is better
        model_wer_normalized = 1 / (1 + model_wer)
        if section != 'full' and section != 'ASR':
            section_wers.append(model_wer)
            section_normalized_wers.append(model_wer_normalized)
        if section == 'full':
            accs["full lyrics transcription"]["WER"] = model_wer
            accs["full lyrics transcription"]["IWER"] = model_wer_normalized
    if section_wers:
        accs["section lyrics transcription"]["WER"] = np.mean(section_wers)
        accs["section lyrics transcription"]["IWER"] = np.mean(section_normalized_wers)
        
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