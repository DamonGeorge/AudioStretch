import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
from output import Output
from input import Input
from loop import AudioLoop
from lib.btrack import BeatTracker  # pylint: disable=import-error,no-name-in-module
import librosa
import pickle
import time


def parse_args():
    """
    Parses command line arguments.
    Args: file, device
    """
    # Helper function for argument parsing.
    def int_or_str(text):
        try:
            return int(text)
        except ValueError:
            return text

    parser = argparse.ArgumentParser(
        description="???"
    )
    parser.add_argument("-l", "--loop", required=True, help="*.pkl file containing AudioLoop object")
    parser.add_argument("-i", "--input", type=int_or_str, help="either input device # or file to stream as input")
    parser.add_argument("-o", "--output", type=int, default=None, help="output device #")
    parser.add_argument("-b", "--block-size", type=int, default=1024, help="audio block size in frames")

    return parser.parse_args()


def create_btrack_callback(btrack: BeatTracker):

    def callback(block: np.ndarray):
        if block.ndim > 1:
            if block.shape[1] > 1:
                block = librosa.to_mono(block.T)
            else:
                block = np.squeeze(block)
        btrack.process_audio(block)

        if btrack.beat_due_in_current_frame():
            print("Beat")

    return callback


def main():
    args = parse_args()
    block_size = args.block_size
    buf_size = block_size*20
    input_buffer = np.zeros((buf_size, 2))
    loop_buffer = np.zeros((buf_size, 2))

    btrack = BeatTracker(hop_size=512, frame_size=block_size)

    input = Input(filename=args.input if isinstance(args.input, str) else None,
                  device=args.input if isinstance(args.input, int) else None,
                  block_size=block_size, input_buffer=input_buffer, callback=create_btrack_callback(btrack))
    output = Output(input_buffer, block_size=block_size, sample_rate=input.sample_rate)

    with open(args.loop, 'rb') as f:
        loop = pickle.load(f)
    loop_output = Output(loop_buffer, block_size=block_size, sample_rate=loop.sample_rate, device=args.output)

    input.start()
    output.start()
    time.sleep(1)
    loop_output.start()

    while True:
        time.sleep(1)


if __name__ == "__main__":
    main()
