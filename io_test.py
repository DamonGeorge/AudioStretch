
"""
This program routes input in real time from either the default input device or given file
to the given output device
"""
import argparse

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa
from input import Input
from output import Output


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


def main():
    args = parse_args()

    filename = args.file
    output_device = args.device
    block_size = args.block_size
    out_idx = 0
    in_idx = 0
    buf_size = 10240
    data = np.zeros((buf_size, 2))

    input = Input(data, filename=filename, block_size=block_size)
    sample_rate = input.sample_rate  # default sample rate

    print(f"Sample rate: {sample_rate}")
    output = Output(data, block_size=block_size, sample_rate=sample_rate)
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


# run main in the event loop
if __name__ == "__main__":
    main()
