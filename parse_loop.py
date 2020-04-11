"""
This is a script to create and save and AudioLoop object
from a given file. This will play the audio with the
detected beats, allowing the user to confirm to save
"""
import argparse
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from loop import AudioLoop
import sounddevice as sd
import pickle


def parse_args():
    """
    Parses command line arguments.
    Args: file
    """
    parser = argparse.ArgumentParser(description="Generate and save an AudioLoop object from the given audio file")
    parser.add_argument("file", type=str, help="audio file to be processed")
    parser.add_argument("-t", "--tempo", type=int, default=120, help="estimated tempo of the audio file")
    parser.add_argument("--hop", type=int, default=512, help="beat detection hop length")
    parser.add_argument("--no-align", action="store_true", help="don't align beats to the beginning of the audio")

    return parser.parse_args()


def main():
    args = parse_args()
    filepath = Path(args.file).resolve()
    if not filepath.exists():
        raise FileNotFoundError()

    loop = AudioLoop(path=filepath, hop_length=args.hop, estimated_bpm=args.tempo,
                     align_beats_to_start=not args.no_align)

    click_track = librosa.core.clicks(frames=loop.beat_frames, sr=loop.sample_rate, length=loop.samples)
    sd.play(librosa.to_mono(loop.audio.T)+click_track)

    val = input("save audio loop? (y/n)")

    sd.stop()
    if val == 'y':
        with open(f"{filepath.stem}_LOOP.pkl", 'wb') as f:
            pickle.dump(loop, f)
        print(f"Saved to {filepath.stem}_LOOP.pkl")

    print("Goodbye!")


if __name__ == '__main__':
    main()