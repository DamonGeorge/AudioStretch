
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
from lib.btrack import BeatTracker  # pylint: disable=import-error,no-name-in-module
from utils.queue_buffer import QueueBuffer
from utils.input_file_stream import InputFileStream


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

    btrack = BeatTracker(hop_size=block_size, frame_size=block_size)

    tempo = 120

    def btrack_callback(block: np.ndarray, frames):
        nonlocal tempo, btrack

        block = block[:frames]

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

    def input_callback(indata, frames, *args, **kwargs):
        btrack_callback(indata, frames)
        buffer.put(indata, frames)

    def output_callback(outdata, frames, *args, **kwargs):
        if not buffer.get_into_nowait(outdata, length=frames):
            outdata[:] = 0

    # select either input stream or file
    if args.input is None or isinstance(args.input, int):
        input = sd.InputStream(samplerate=sample_rate, blocksize=block_size,
                               device=args.input, dtype="float32", callback=input_callback)
    elif isinstance(args.input, str):
        input = InputFileStream(args.input, block_size=block_size, callback=input_callback)
        sample_rate = input.sample_rate  # default sample rate

    print(f"Sample rate: {sample_rate}")

    output = sd.OutputStream(blocksize=block_size, samplerate=sample_rate,
                             device=output_device, callback=output_callback)

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
