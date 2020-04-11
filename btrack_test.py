
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
import asyncio
from input import Input
from output import Output
from lib.btrack import BeatTracker  # pylint: disable=import-error,no-name-in-module


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


async def main():
    args = parse_args()

    filename = args.file
    output_device = args.device
    block_size = args.block_size
    buf_size = 10240
    data = np.zeros((buf_size, 2))
    btrack = BeatTracker(hop_size=512, frame_size=block_size)

    input = Input(data, filename=filename, block_size=block_size, callback=create_btrack_callback(btrack))
    sample_rate = input.sample_rate  # default sample rate

    print(f"Sample rate: {sample_rate}")
    output = Output(data, block_size=block_size, sample_rate=sample_rate)
    try:
        print(f"Input Starting: {time.perf_counter()}")
        await input.start()
        print(f"Output Starting: {time.perf_counter()}")
        output.start()

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print('\nInterrupted by user')

    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

    finally:
        output.stop()
        input.stop()


# run main in the event loop
asyncio.run(main())