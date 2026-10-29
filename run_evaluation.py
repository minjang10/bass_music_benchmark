import os
import ast
import re
import random
import argparse
import json

from datasets import Audio, load_dataset

CATEGORIES = ['structural_segmentation', 'lyrics_transcription', 'artist_collaboration', 'musicological_analysis']
MCQ_TASKS = ['Single-Gene Detection', 'Pairwise-Gene Detection', 'ranking_genes']

# Task names in the Hugging Face release -> (task name read by src/metrics/*, answer type)
COLLAB_TASKS = {
    'Standard Artist Counting': ('artist_count', int),
    'artist_count_feature': ('artist_count_feature', int),
    'Temporal Artist Counting': ('timestamp_artist_count', int),
    'Section Artist Counting': ('section_artist_count', int),
    'Vocal Delivery Counting': ('delivery_count', int),
    'Target Artist Duration': ('artist_duration', float),
    'Vocal Delivery Duration': ('delivery_duration', float),
    'Artist Delivery Duration': ('artist_delivery_duration', float),
    'Section Duration': ('section_duration', float),
    'Artist Localization': ('identify_artist_start_timestamp', float),
    'Sectional Artist Role': ('section_function_identification', str),
    'Sectional Vocal Style': ('section_delivery_identification', str),
    'Temporal Vocal Style': ('timestamp_delivery_identification', str),
    'Vocal Delivery Style Comparison': ('delivery_comparison', str),
}
MUSICOLOGY_TASKS = {
    'Single-Gene Detection': 'multiple choice question',
    'Pairwise-Gene Detection': 'pair selection',
    'ranking_genes': 'ranking_genes',
    'Gene Attribution': 'multi_song_mcq_max',
}


def literal(x):
    # Some columns store Python lists/dicts as strings
    return ast.literal_eval(x) if isinstance(x, str) else x


def run_inference(model, question, audio):
    # Placeholder for actual inference code
    # This function should take the model, question, and audio as input
    # and return the model's prediction.
    # `audio` is {'path', 'bytes'} (a WAV) for structural_segmentation and a list of YouTube URLs otherwise.
    # Gene Attribution lists four songs; the paper concatenates them with 5 s of silence in between.
    prediction = "predicted_answer"  # Replace with actual prediction logic
    return prediction


def first_section(answer):
    # Section answers are a list of {'section', ...} occurrences or a single one
    gt = literal(answer)
    return (gt[0] if isinstance(gt, list) else gt)['section']


def get_prompt(question):
    prompt = question['question']
    if question['task'] == 'Section Structural Lyrics Transcription':
        # The released prompts still contain the unformatted template; use the placeholder of the paper's Table 5
        prompt = prompt.replace('{{', '{').replace('}}', '}').replace('{section}', 'section_name')
    return prompt


def to_metric_format(question, category):
    # Adds the `task`, `gt` (and `section`, `answer_choices`) fields that src/metrics/* read
    task, answer = question['task'], question['answer']
    output = {'bass_task': task}
    if category == 'structural_segmentation':
        output['gt'] = literal(answer)
        if task.startswith('Section'):
            output['task'] = 'section_structural_segmentation'
            output['section'] = first_section(answer)
        else:
            output['task'] = 'full_structural_segmentation'
    elif category == 'lyrics_transcription':
        output['gt'] = literal(answer)
        if task.startswith('Full'):
            output['task'] = 'full lyrics transcription'
        else:
            output['task'] = re.sub(r'[0-9]+$', '', first_section(answer)) + ' lyrics transcription'
    elif category == 'artist_collaboration':
        output['task'], answer_type = COLLAB_TASKS[task]
        output['gt'] = answer_type(answer)
    else:
        output['task'] = MUSICOLOGY_TASKS[task]
        output['gt'] = answer
        output['answer_choices'] = list(get_choices(question)) or literal(question['answer_choices'])
    return output


def get_choices(question):
    # {choice name: description}; `answer_choices` holds descriptions instead of names in a few rows
    choices = literal(question['answer_choices_with_descriptions'])
    return literal(choices[0]) if choices else {}


def run_evaluation(model, category, dataset):
    outputs = []

    # Load dataset based on category
    questions = load_dataset(dataset, category, split='test')
    if category == 'structural_segmentation':
        questions = questions.cast_column('audio', Audio(decode=False))

    # Process each question
    for question in questions:
        prompt = get_prompt(question)
        audio = question['audio'] if category == 'structural_segmentation' else literal(question['youtube_url'])

        # Handle MCQ tasks with shuffled answer choices.
        # Shuffling is required: the gold answer is always listed first in the released choices.
        if question['task'] in MCQ_TASKS:
            preds = []
            answer_choices_descriptions = [f"{ac}: {desc}" for ac, desc in get_choices(question).items()]

            for _ in range(4):
                random.shuffle(answer_choices_descriptions)
                current_question = prompt + "\nAnswer choices:\n" + "\n".join(answer_choices_descriptions)
                prediction = run_inference(model, current_question, audio)
                preds.append(prediction)

            prediction = preds
        else:
            prediction = run_inference(model, prompt, audio)

        output = {k: v for k, v in question.items() if k != 'audio'}
        output.update(to_metric_format(question, category))
        output['prompt'] = prompt
        output['prediction'] = prediction
        outputs.append(output)

    return outputs


def main():
    parser = argparse.ArgumentParser(description="Evaluate model on BASS")
    parser.add_argument('--category', type=str, choices=CATEGORIES + ['all'], required=True,
                        help='Category of tasks to evaluate')
    parser.add_argument('--output-dir', type=str, default='evaluation_results', help='Directory to save evaluation results')
    parser.add_argument('--dataset', type=str, default='oreva/bass_music_benchmark',
                        help='Hugging Face dataset name or a local copy of it')
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Replace with actual model loading code
    model = "Your Model"

    random.seed(0)
    categories = CATEGORIES if args.category == 'all' else [args.category]
    for category in categories:
        # Evaluate
        outputs = run_evaluation(model, category, args.dataset)

        # Save outputs
        output_file = os.path.join(args.output_dir, f'{category}_output.json')
        with open(output_file, 'w') as f:
            json.dump(outputs, f, indent=4)


if __name__ == "__main__":
    main()
