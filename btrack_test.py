
"""
This program detects beats in real time from the input source 
(either a file or input device)
"""
import argparse

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa
from input import Input
from output import Output
from lib.btrack import BeatTracker  # pylint: disable=import-error,no-name-in-module
from circular_buffer import CircularBuffer


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

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", help="audio file to be streamed as input")
    parser.add_argument("-d", "--device", type=int_or_str, help="output device")
    parser.add_argument("-b", "--block-size", type=int, default=1024, help="audio block size in frames")

    return parser.parse_args()


def create_btrack_callback(btrack: BeatTracker):
    tempo = 120

    def callback(block: np.ndarray):
        nonlocal tempo

        if block.ndim > 1:
            if block.shape[1] > 1:
                block = librosa.to_mono(block.T)
            else:
                block = np.squeeze(block)
        btrack.process_audio(block)

        if btrack.beat_due_in_current_frame():
            print("Beat")

            new_tempo = btrack.get_current_tempo_estimate()
            if new_tempo != tempo:
                tempo = new_tempo
                print(f"Tempo: {tempo}")

    return callback


def main():
    args = parse_args()

    filename = args.file
    output_device = args.device
    block_size = args.block_size
    buf_size = 10240
    buffer = CircularBuffer((buf_size, 2))
    btrack = BeatTracker(hop_size=512, frame_size=block_size)
    btrack.fix_tempo(120)

    input = Input(input_buffer=buffer, filename=filename, block_size=block_size,
                  callback=create_btrack_callback(btrack))
    sample_rate = input.sample_rate  # default sample rate

    print(f"Sample rate: {sample_rate}")
    output = Output(buffer, block_size=block_size, sample_rate=sample_rate)
    try:
        print(f"Input Starting: {time.perf_counter()}")
        input.start()
        print(f"Output Starting: {time.perf_counter()}")
        output.start()

        while True:
            time.sleep(1)

    except KeyboardInterrupt:
        print('\nInterrupted by user')

    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

    finally:
        output.stop()
        input.stop()


if __name__ == "__main__":
    main()
