import os
import subprocess
import time
import argparse
import sys
import psutil
import gc
from datetime import datetime

from lnco_transcribe.utils.format_helpers import get_files, convert_str_to_csv
from lnco_transcribe.utils.preprocessing_helpers import preprocessing_csv

def log_memory_usage(context=""):
    process = psutil.Process()
    rss = process.memory_info().rss / 1024**2  # in MB
    print(f"[MEM] After {context}: RSS={rss:.1f} MB")

def write_log(message, log_path="processing_log.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"[{timestamp}] {message}\n")


def process_audio_file(audio_file, directory, whisper_model, language, task=None, overwrite=False):
    """Process a single audio file with the diarization script."""

    # Determine the output file paths
    experiment_name = os.path.basename(os.path.normpath(directory))  # Extract only the last folder name
    relative_path = os.path.relpath(audio_file, directory)

    if task == "translate":
        output_dir = os.path.join("results", experiment_name, f"{language}_to_eng", os.path.dirname(relative_path))
    else:
        output_dir = os.path.join("results", experiment_name, os.path.dirname(relative_path))

    os.makedirs(output_dir, exist_ok=True)  # Ensure the output directory exists

    base_name = os.path.splitext(os.path.basename(audio_file))[0] # Get the file name without extension
    str_file = os.path.join(output_dir, f"{base_name}.str")
    csv_file = str_file.replace(".str", ".csv")

    # Check if transcription already exists
    if not overwrite and os.path.exists(csv_file):
        print(f"Skipping {audio_file}: {csv_file} already exists.")
        return None
    
    print(f"Processing {audio_file}...")
    write_log(f"Started processing {audio_file}")

    # Start the timer
    start_time = time.time()

    command = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), 'whisper_diarization', 'diarize.py'),
        "-a", audio_file,
        "-d", output_dir,
        "--whisper-model", whisper_model,
        "--language", language,
        "--task", task,
    ]

    # Run the Python script
    subprocess.run(command)

    # End the timer
    end_time = time.time()
    elapsed_time = end_time - start_time
    finished_message = f"Finished processing {audio_file} in {int(elapsed_time // 60)} min and {elapsed_time % 60:.0f} sec"
    print(f"\n\n {finished_message}")
    write_log(finished_message)
    # Convert .str file to .csv format
    convert_str_to_csv(str_file, experiment_name)

    return str_file

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process audio files for diarization.")
    parser.add_argument(
        "-d", "--directory",
        type=str,
        required=True,
        help="Directory containing audio files."
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="large-v3",
        help="Whisper model to use."
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="The language spoken in the audio. If the detected language is different "
            "from the specified language and 'task'=None or 'transcribe', "
            "the model will translate the audio to the specified language, otherwise if task='translate' it will translate the audio to English."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="Task to execute (transcribe or translate). Specify None to follow the 'language' argument; "
            "it will translate when the audio does not match the specified language, useful for multilingual audios. "
            "If the audio is entirely in another language and you want to translate to English (Whisper's best performance), "
            "you can use the 'translate' task."
    )
    parser.add_argument(
    "--also-transcribe",
    action="store_true",
    help="When task is 'translate', also perform a transcription in the original language."
    )
    parser.add_argument(
        "-e", "--extensions",
        type=str,
        nargs='+', # can give multiples arguments separate by a space
        default=[".m4a",".mp4",".wav"],
        help="List of allowed audio file extensions."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If specified, overwrite existing transcriptions."
    )
    args = parser.parse_args()

    # Get audio files
    audio_files = get_files(args.directory, args.extensions)
    print("Parse dir: ", args.directory)
    print("Parse audio: ", audio_files)
    print("Parse extensions: ", args.extensions)
    print("Parse Language: ", args.language)
    print("Parse Task: ", args.task)

    # Process each audio file
    str_files = []
    for audio_file in audio_files:
        # Step 1: (Optional) Transcribe in original language before translating
        if args.task == "translate" and args.also_transcribe:
            print(f"\nAlso transcribing {audio_file} in original language ({args.language})")
            str_file_transcribe = process_audio_file(audio_file, args.directory, args.whisper_model, args.language, 
                                                     task="transcribe", overwrite=args.overwrite)
            log_memory_usage(f"transcribe {audio_file}")

            if str_file_transcribe:
                str_files.append(str_file_transcribe)
                # Preprocessing version of the csv, inside `results/processed`
                preprocessing_csv(str_file_transcribe)
        time.sleep(2)  # Give OS time to clean up
        gc.collect()

        # Step 2: Main task (either transcribe or translate)
        str_file = process_audio_file(audio_file, args.directory, args.whisper_model, args.language, 
                                      task="translate" if args.task == "translate" else args.task, overwrite=args.overwrite)
        log_memory_usage(f"translate {audio_file}")
        if str_file:
            str_files.append(str_file)
            # Preprocessing version of the csv, inside `results/processed`
            preprocessing_csv(str_file)

if __name__ == "__main__":
    main()