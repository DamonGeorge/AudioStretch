
"""
This program routes input in real time from either the default input device or given file
to the given output device
"""
from utils.input_file_stream import InputFileStream
from utils.queue_buffer import QueueBuffer
import librosa
import time
import numpy as np
import soundfile as sf
import sounddevice as sd
import argparse


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
    parser.add_argument("-i", "--input", type=int_or_str, default=None,
                        help="either input device # or file to stream as input")
    parser.add_argument("-d", "--device", type=int_or_str, help="output device")
    parser.add_argument("-b", "--block-size", type=int, default=1024, help="audio block size in frames")

    return parser.parse_args()


def main():
    args = parse_args()

    output_device = args.device
    block_size = args.block_size
    buf_size = 10240
    buffer = QueueBuffer((buf_size, 2))

    sample_rate = 44100

    def input_callback(indata, frames, *args, **kwargs):
        buffer.put(indata, frames)

    def output_callback(outdata, frames, *args, **kwargs):
        if not buffer.get_into_nowait(outdata, length=frames):
            outdata[:] = 0

    # select either input stream or file
    if args.input is None or isinstance(args.input, int):
        input = sd.InputStream(samplerate=sample_rate, blocksize=block_size, latency='low',
                               device=args.input, dtype="float32", callback=input_callback)
    elif isinstance(args.input, str):
        input = InputFileStream(args.input, block_size=block_size, callback=input_callback)
        sample_rate = input.sample_rate  # default sample rate
    print(f"Sample rate: {sample_rate}")

    output = sd.OutputStream(blocksize=block_size, samplerate=sample_rate, latency='low',
                             device=output_device, callback=output_callback)
    try:
        print(f"Input Starting: {time.perf_counter()}")
        input.start()
        print(f"Output Starting: {time.perf_counter()}")
        output.start()

        while True:
            time.sleep(1)
            print(f"input latency: {input.latency}")
            print(f"output latency: {output.latency}")

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
