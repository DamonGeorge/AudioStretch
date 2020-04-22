
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
from circular_buffer import CircularBuffer
from aubio import tempo


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


def create_tracker_callback(beat_tracker):
    tempo = 120

    def callback(block: np.ndarray):
        nonlocal tempo

        if block.ndim > 1:
            if block.shape[1] > 1:
                block = librosa.to_mono(block.T)
            else:
                block = np.squeeze(block)

        is_beat = False
        for i in range(2):
            bt = beat_tracker(block[i*512:(i+1)*512])
            is_beat = is_beat or bt
        new_tempo = beat_tracker.get_bpm()
        if new_tempo != tempo:
            tempo = new_tempo
            print(f"Tempo: {tempo}")
        if is_beat:
            print("Beat")

    return callback


def main():
    args = parse_args()

    filename = args.file
    output_device = args.device
    block_size = args.block_size
    buf_size = 10240
    buffer = CircularBuffer((buf_size, 2))
    if filename:
        input_sample_rate = sf.SoundFile(filename).samplerate
    else:
        input_sample_rate = 44100

    beat_tracker = tempo(buf_size=1024, samplerate=input_sample_rate)

    input = Input(input_buffer=buffer, filename=filename, block_size=block_size,
                  callback=create_tracker_callback(beat_tracker))

    print(f"Sample rate: {input_sample_rate}")
    output = Output(buffer, block_size=block_size, sample_rate=input_sample_rate)
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
