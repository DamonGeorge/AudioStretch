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
from circular_buffer import CircularBuffer
from lib.rubberband import AudioStretcher  # pylint: disable=import-error,no-name-in-module
from threading import Thread


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
    parser.add_argument("-l", "--loop", required=True, help="*.pkl AudioLoop file to stream as input")
    parser.add_argument("-o", "--output", type=int, default=None, help="output device #")
    parser.add_argument("-b", "--block-size", type=int, default=1024, help="audio block size in frames")

    return parser.parse_args()


def main():
    args = parse_args()
    block_size = args.block_size
    buf_size = block_size*20
    output_buffer = CircularBuffer((buf_size, 2))
    out_idx = 0
    processed_samples = 0

    with open(args.loop, 'rb') as f:
        loop: AudioLoop = pickle.load(f)
    loop_output_stream = Output(output_buffer, block_size=block_size, sample_rate=loop.sample_rate, device=args.output)

    stretcher = AudioStretcher(sample_rate=loop.sample_rate, channels=loop.channels, realtime=True)

    time_scale = 1.1

    input("Press enter to start loop playback")

    while processed_samples <= buf_size - 2 * block_size:
        # time_scale = loop.tempo / current_tempo
        stretched = stretcher.stretch(loop.get_next_block(block_size), time_scale)
        if stretched.shape[0] > 0:
            out_idx = output_buffer.put(out_idx, stretched)
            processed_samples += np.shape(stretched)[0]
            print(f"stretched samples: {processed_samples}")

    print("OUTPUT STARTING")
    loop_output_stream.start()

    while True:
        # time_scale = loop.tempo / current_tempo
        start = time.perf_counter()

        stretcher.set_time_ratio(time_scale)
        stretcher.process(loop.get_next_block(block_size), False)

        stretched = stretcher.retrieve()
        while stretched.shape[0] > 0:
            processed_samples += np.shape(stretched)[0]
            next_idx = output_buffer.get_next_idx(out_idx, np.shape(stretched)[0])
            while (processed_samples - loop_output_stream.processed_samples) >= buf_size:
                print("sleep")
                time.sleep(0.001)
            out_idx = output_buffer.put(out_idx, stretched)
            stretched = stretcher.retrieve()

        print(f"Loop time: {time.perf_counter() - start}")


if __name__ == "__main__":
    main()
