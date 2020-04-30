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
from utils.audio_loop import AudioLoop
import sounddevice as sd


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
    parser.add_argument("--num-beats", type=int, default=None, help="predefined number of beats in track")
    return parser.parse_args()


def main():
    args = parse_args()
    filepath = Path(args.file).resolve()
    if not filepath.exists():
        raise FileNotFoundError()

    loop = AudioLoop(path=filepath, hop_length=args.hop, estimated_bpm=args.tempo,
                     align_beats_to_start=not args.no_align)

    if args.num_beats is not None:
        print(f"Using given num_beats to compute the tempo and beat times")
        samples_per_beat = loop.samples / args.num_beats
        loop.beat_samples = np.rint(np.linspace(0, samples_per_beat*(args.num_beats-1), args.num_beats))
        loop.beat_frames = np.rint(loop.beat_samples / args.hop)

    print(f"tempo: {loop.tempo}")

    click_track = librosa.core.clicks(frames=loop.beat_frames, sr=loop.sample_rate, length=loop.samples)
    sd.play(librosa.to_mono(loop.audio.T)+click_track)

    val = input("save audio loop? (y/n)")

    sd.stop()
    if val == 'y':
        loop.save(f"{filepath.stem}_LOOP.pkl")
        print(f"Saved to {filepath.stem}_LOOP.pkl")

    print("Goodbye!")


if __name__ == '__main__':
    main()
