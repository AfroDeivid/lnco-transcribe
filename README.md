# lnco-transcribe: A CLI Tool for Automated Audio Transcription & Diarization

A lightweight and efficient CLI tool for converting audio interviews into structured text, featuring speaker diarization and multi-language support.

This repository is a packaged version of the [Automatic Interviews Processing project](https://github.com/AfroDeivid/automatic-interviews-processing).
The original project covers a broader workflow, including text preprocessing, transcript evaluation, text analysis, and topic modeling. However, this repository focuses exclusively on the **Audio-to-Text** module.

Developed by [David Friou](https://github.com/AfroDeivid) as part of a semester project at [LNCO Lab](https://www.epfl.ch/labs/lnco/).

![Project Workflow](images/WD_pipeline.png)

## Table of Contents
1. [Installation](#installation)  
2. [Usage](#usage)  
   1. [Preparing Your Data](#preparing-your-data)  
   2. [Transcription & Diarization (Audio-to-Text)](#transcription--diarization-audio-to-text)  
   3. [Outputs](#outputs)  
3. [File Structure](#file-structure)  
4. [Mentions](#mentions)

# Installation

## 1. Prerequisites
Before installing the package, ensure you have the following dependencies installed:

- Install ``FFMPEG`` from [here](https://ffmpeg.org/download.html), you can follow a guide like [this](https://phoenixnap.com/kb/ffmpeg-windows) for Windows installation.  
Ensure that FFMPEG is added to your system’s PATH.

- Install ``Strawberry Perl`` from [here](https://strawberryperl.com/).

- ``Visual C++ Build Tools``: If you encounter build errors during installation, install the Visual C++ Build Tools by following this [guide](https://stackoverflow.com/questions/40504552/how-to-install-visual-c-build-tools) or download directly from [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).

- Python >= ``3.10`` is required.

## 2. Installation Options

###  Option 1: Install with ``Poetry``

This ensures all dependencies are resolved and the CLI (`lnco-transcribe`) is installed correctly.

- If you don’t have Poetry installed, install it globally:

```
 pip install poetry
``` 
- Install the project and dependencies:
```
poetry install
```

*When using Poetry, the CLI is only available inside the environment. Remember to ``use poetry run`` or ``poetry shell``.*

### Option 2: Install Locally with pip

Create an environment and install the package from source (same effect as Poetry, but without Poetry itself):

```
pip install .
```

### Option 3: Manual Dependency Install

If you want to manually manage dependencies, create an environment and install dependencies (the CLI won’t be available) :

**Locked Environment Installation**  
This setup recreates the *exact environment* used:
```
conda create --name tti python=3.10 --yes
conda activate tti
pip install -r freeze_requirements.txt
``` 

**Flexible / Adaptive Installation**  
If you need more flexibility, like updating certain packages or adapting the repository replace the ``pip install -r freeze_requirements.txt`` step with:
```
pip install cython
pip install -c constraints.txt -r requirements.txt 
``` 

## Running the CLI Tool

Depending on your install method:

- If installed via ``Poetry`` or with ``pip install .`` :

You can run the CLI directly using:

```
poetry run lnco-transcribe -d path_to_folder --whisper-model large-v3 --language en
```

- If installed manually via ``requirements.txt`` :

The CLI entry point won’t be available. Run the module directly instead:

```
python -m lnco_transcribe.run_diarize -d path_to_folder --whisper-model large-v3 --language en
```

*All CLI examples below assume that the package has been installed using ``Poetry`` or via ``pip install .`` and that you're inside an activated environment.*

# Usage

## Preparing Your Data

The pipeline supports nested folder structures, making it easy to process multiple experiments and interviews. To use the pipeline:

- Simply indicate the path to your folder with your audio files.
- The pipeline recursively processes all audio files within these folder and subfolders.

## Transcription & Diarization (Audio-to-Text)
To transcribe audio files with speaker diarization, use:

- **Transcribe audio in its original language :** *(specified with --language)* 
  
```bash
lnco-transcribe -d path_to_folder --whisper-model large-v3 --language en
```

- **Transcribe and translate audio to english :** *(e.g. from french to english)*
```bash
lnco-transcribe -d path_to_folder --whisper-model large-v3 --language fr --task translate
```
- Optionally add ``--also-transcribe`` to also save a transcript in the original language alongside the translated one. Outputs will be saved to:
  - ``results/<experiment>/<subfolders...>`` for original transcription
  - ``results/<experiment>/fr_to_eng/<subfolders...>`` for translated version

If only ``language`` is specified, the model will attempt to translate any detected language into the specified language.

To improve performance, specify the task as ``translate`` if you know in advance that the audio is in a certain language (e.g., French) and want to translate it into English.

- You can view the list of all supported languages along with their corresponding language codes just here: [Languages](lnco-transcribe/whisper_diarization/helpers.py)

| Parameter         | Description                                         | Default                         |
|-------------------|-----------------------------------------------------|---------------------------------|
| **`-d, --directory`** | Path to the directory containing audio files.       | None                 |
| **`--whisper_model`** | Name of the Whisper model used for transcription.   | None                      |
| **``--language ``**       | Language code for transcription (e.g., `fra` for French, `eng` for English). | None                            |
| **``--task ``**           | Task to perform (e.g., "transcribe", "translate").  | None                            |
| **``--also-transcribe``**           | When using ``--task translate``, also transcribes the audio in the original language before translation.  | False                            |
| **``-e, --extensions``**      | List of allowed audio file extensions.              | [".m4a", ".mp4", ".wav"]        |
| **``--overwrite``**       | Overwrites existing transcriptions if specified.    | False                           |

*Run ``lnco-transcribe --help`` for full options &/or see [run_diarize.py](lnco-transcribe/run_diarize.py) for additional information.*

## Outputs
The tool generates transcripts in two structured formats:

- **Text Format:** Simplified and easy-to-read files for manual review.
- **CSV Format:** A structured format ideal for analysis, with columns such as:
  - Experiment name (derived from the name of the folder directory).
  - File name.
  - Participant ID.
  - Timestamps for each segment.
  - Speaker roles and transcription content.

### Processed folder
Contains the same outpouts after aditional preprocessing steps:

- Removal of vocalized fillers
- Visual cleaning of the text
- Prediction of the speaker role in interview set-up (Participant & Interviewer)

For a more modular approach you can use [preprocessing notebook](./lnco-transcribe/preprocessing.ipynb).

# File Structure

## Audio-to-Text Processing  
This section focuses on converting raw audio data into text through transcription and diarization, enabling subsequent analysis.  

- **Preprocessing and Conversion:**  
  - [pre_analysis.ipynb](lnco-transcribe/pre_analysis.ipynb): Analyzes audio files and experiment structure.  
  - [videos_to_audio.py](lnco-transcribe/MTS_to_audio.py): Converts videos formats (`.MTS`,`.mp4`) into `.wav` format for processing.  

- **Transcription & Diarization:**  
  - [run_diarize.py](lnco-transcribe/run_diarize.py): The main script for batch-processing transcription and speaker diarization.  
  - [whisper_diarization/](lnco-transcribe/whisper_diarization/): Source code from the Whisper-Diarization framework. (See [Mentions](#mentions))  
  - [nemo_msdd_configs/](nemo_msdd_configs/): YAML configuration files for diarization tasks.  

- **Transcript Preprocessing:**  
  - [preprocessing.ipynb](lnco-transcribe/preprocessing.ipynb): Modular workflow for cleaning and preparing transcripts for further analysis.  

``utils/`` [format_helpers.py](lnco-transcribe/utils/format_helpers.py) and [preprocessing_helpers.py](lnco-transcribe/utils/preprocessing_helpers.py): Assist with structured formatting and transcript preprocessing.  

# Mentions

This package relies heavily on the **Whisper-Diarization** framework to handle transcription and diarization of audio files into structured text formats, which is licensed under the BSD 2-Clause License.

```bibtex
@unpublished{hassouna2024whisperdiarization,
  title={Whisper Diarization: Speaker Diarization Using OpenAI Whisper},
  author={Ashraf, Mahmoud},
  year={2024}}
```
For additional details, visit the [Whisper-Diarization GitHub repository](https://github.com/MahmoudAshraf97/whisper-diarization).
