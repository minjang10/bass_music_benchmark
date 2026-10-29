import subprocess
import argparse
import os
import sys

METRICS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src', 'metrics')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--structural-segmentation', type=str, required=False)
    parser.add_argument('--musicological-analysis', type=str, required=False)
    parser.add_argument('--artist-collab', type=str, required=False)
    parser.add_argument('--lyrics-transcription', type=str, required=False)
    args = parser.parse_args()
    result_files = [
        ('collab_analysis_evaluation.py', args.artist_collab),
        ('structural_segmentation_evaluation.py', args.structural_segmentation),
        ('musicological_analysis_evaluation.py', args.musicological_analysis),
        ('lyrics_transcription_evaluation.py', args.lyrics_transcription),
    ]

    # Run the evaluation scripts for the result files given
    for script, result_file in result_files:
        if result_file is None:
            continue
        subprocess.run([sys.executable, os.path.join(METRICS_DIR, script), '--result-file', result_file])
        print('=' * 100)


if __name__ == '__main__':
    main()
