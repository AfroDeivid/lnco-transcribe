import os
import shutil
import subprocess
import argparse

def convert_and_backup(raw_root_dir, backup_dir):
    input_directory = raw_root_dir
    valid_extensions = [".mts", ".mp4"] # Video file extensions to look for

    for subdir, _, files in os.walk(input_directory):
        print(f"Scanning: {subdir}")
        for file in files:
            if os.path.splitext(file)[1].lower() in valid_extensions:
                print(f"Matched video: {file}")
                # Determine relative path and construct destination paths
                relative_path = os.path.relpath(subdir, input_directory)
                input_path = os.path.join(subdir, file)

                # Output .wav in corresponding subdirectory under raw_root_dir
                wav_output_dir = os.path.join(raw_root_dir, relative_path)
                os.makedirs(wav_output_dir, exist_ok=True)
                output_filename = f"{os.path.splitext(file)[0]}.wav"
                output_path = os.path.join(wav_output_dir, output_filename)

                # Run ffmpeg
                command = [
                    "ffmpeg", "-i", input_path,
                    "-vn", "-acodec", "pcm_s16le", "-ar", "44100", "-ac", "2",
                    output_path
                ]
                subprocess.run(command)
                print(f"Converted: {input_path} -> {output_path}")

                # Move original to backup
                backup_subdir = os.path.join(backup_dir, relative_path)
                os.makedirs(backup_subdir, exist_ok=True)
                backup_path = os.path.join(backup_subdir, file)
                shutil.move(input_path, backup_path)
                print(f"Moved to backup: {backup_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert .MTS and .MP4 files to .wav and back up originals.")
    parser.add_argument(
        "--raw_root_dir",
        type=str,
        default="./data/grief",
        help="Path to the root directory containing video files (default: ./data/grief)"
    )
    parser.add_argument(
        "--backup_dir",
        type=str,
        default="./data/videos_BACKUP",
        help="Path to the directory where original video files will be moved (default: ./data/MTS_BACKUP)"
    )
    args = parser.parse_args()

    convert_and_backup(args.raw_root_dir, args.backup_dir)