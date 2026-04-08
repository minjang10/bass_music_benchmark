# &#119074; BASS: Benchmarking Audio LMs for Musical Structural and Semantic Reasoning

This repo contains the evaluation code for BASS benchmark for the paper [BASS](https://arxiv.org/abs/2602.04085)

<p style="text-align:center;">Download the BASS Dataset on Hugging Face <a href="https://huggingface.co/datasets/oreva/bass_music_benchmark">🤗</a></p>

## BASS: Benchmarking Audio LMs for Musical Structure and Semantic Reasoning
Music understanding is a complex task that often requires reasoning over both structural and semantic elements of audio. We introduce BASS, designed to evaluate music understanding and reasoning in audio language models across four broad categories: structural segmentation, lyric transcription, musicological analysis, and artist collaboration. BASS comprises 2658 questions spanning 12 tasks, 1993 unique songs and covering over 138 hours of music from a wide range of genres and tracks, crafted to assess musicological knowledge and reasoning in real-world scenarios. We evaluate 14 open-source and frontier multimodal LMs, finding that even state-of-the-art models struggle on higher-level reasoning tasks such as structural segmentation and artist collaboration, while performing best on lyric transcription. Our analysis reveals that current models leverage linguistic priors effectively but remain limited in reasoning over musical structure, vocal, and musicological attributes. BASS provides an evaluation framework with widespread applications in music recommendation and search and has the potential to guide the development of audio LMs.

![BASS Figure](assets/main_figure.png)

NB: This data should only be used for evaluation purposes and not for model training.

## Tasks Covered in BASS
### Structural Segmentation
- **Full Structural Segmentation**: Provide the start and end timestamps, along with the name of every musical section (e.g., intro, verse, chorus) of the song.
- **Section Structural Segmentation**: Provide the start and end timestamps of a specific section given the section name.

### Structural Lyric Transcription
- **Full Structural Lyrics Transcription**: Segment the full song into its structural sections and transcribe the lyrics of each section
- **Section Structural Lyrics Transcription**: Segment a specific musical section of a song and transcribe only those lyrics. 

### Musicicological Analysis
- **Single-Gene Detection**: Given four musicological attributes of a song and their descriptions, identify the most dominant attribute.
- **Pairwise-Gene Detection**: Given eight musicological attributes of a song and their descriptions, identify the pair of attributes that are the most dominant.
- **Gene Attribution**: Given a single musicological attribute and its description, identify which recording expresses this attribute most prominently given four options.
- **Gene Dominance Ranking**: Given four musicological attributes of a song and their descriptions, rank the attributes in ascending order of dominance in the song.

### Artist Collaboration
- **Artist Counting**:
  - Standard Artist Counting: Report the total number of artists performing in a song.
  - Featured Artist Counting: Report the total number of featured artists in a song.
  - Vocal Delivery Counting: Report the total number of artists using a specific vocal delivery (singing, rapping, spoken) in a song.
  - Section Artist Counting: Report the total number of artists within a specific musical section of the song. 
  - Temporal Artist Counting: Report the total number of artists withing a start and end timestamps.
- **Artist Duration**:
  - Target Artist Duration: Report the total duration of a specific target artist.
  - Vocal Delivery Duration: Report the total duration of a specific vocal delivery, regardless of which artist is performing it.
  - Artist Delivery Duration: Report the total duration of a specific target artist performing the vocal delivery.
  - Section Duration: Report the total duration of a specific section.
- **Artist Localization**: Report the starting timestamp of when a target artist first appears in the song.
- **Artist Attribution**:
  - **Vocal Style Comparison**: Determine if the vocal delivery between two artists is the same or different.
  - **Sectional Vocal Style**: Identify the delivery style of a specific artist in a section.
  - **Sectional Artist Role**: Identify the role (main artist, featured artist, or neither) of an artist in a section.
  - **Temporal Vocal Style**: Identify the delivery style within the given start and end timestamps.


## Download the data
```
from huggingface_hub import snapshot_download

# Download data for all tasks
snapshot_download(repo_id="oreva/bass", repo_type="dataset", allow_patterns="*.json", local_dir=".")
```


## Run Evaluation
Change `model_name` in `run_evaluation.py` to your model
```
python run_evaluation.py --category all --output-dir results
```
#### Single Task Evaluation
```
python run_evaluation.py --category structural-segmentation --output-dir results
```

## Compute Performance
```
python run_metrics.py \
  --structural-segmentation results/structural_segmentation_output.json \
  --structural-lyrics-transcription results/structural_lyrics_transcription_output.json \
  --artist-collab results/artist_collaboration_output.json \
  --musicological-analysis results/musicological_analysis.json
```
#### To compute the performance of a single task:
```
python collab_analysis_evaluation.py \
  --result_file results/artist_collaboration_output.json
```
Note: Due to the length of the lyrics transcription outputs, computing the metrics may take a long time. To mitigate this, the script we provide for lyrics transcription processes the results concurrently, which may take up compute. 

## Contact
If you have any questions, please feel free to contact us via oahia@cs.washington.edu or minjang@cs.washington.edu.

## Citation
```
@misc{jang2026bassbenchmarkingaudiolms,
      title={BASS: Benchmarking Audio LMs for Musical Structure and Semantic Reasoning}, 
      author={Min Jang and Orevaoghene Ahia and Nazif Tamer and Sachin Kumar and Yulia Tsvetkov and Noah A. Smith},
      year={2026},
      eprint={2602.04085},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2602.04085}, 
}
```
