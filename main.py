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
    parser.add_argument("-l", "--loop", required=True, help="*.pkl file containing AudioLoop object")
    parser.add_argument("-i", "--input", type=int_or_str, help="either input device # or file to stream as input")
    parser.add_argument("-o", "--output", type=int, default=None, help="output device #")
    parser.add_argument("-b", "--block-size", type=int, default=1024, help="audio block size in frames")

    return parser.parse_args()


def main():
    args = parse_args()
    block_size = args.block_size
    buf_size = block_size*20
    input_buffer = CircularBuffer((buf_size, 2))
    loop_buffer = CircularBuffer((buf_size, 2))
    loop_idx = 0
    processed_samples = 0
    input_sample_rate = 44100

    btrack = BeatTracker(hop_size=512, frame_size=block_size)
    current_tempo = 120

    def btrack_callback(block: np.ndarray):
        nonlocal btrack, current_tempo, input_sample_rate

        if block.ndim > 1:
            if block.shape[1] > 1:
                block = librosa.to_mono(block.T)
            else:
                block = np.squeeze(block)
        btrack.process_audio(block)

        if btrack.beat_due_in_current_frame():
            print("Beat")

        tempo = btrack.get_current_tempo_estimate() * (input_sample_rate / 44100)
        if abs(tempo - current_tempo) > 0.1:
            current_tempo = tempo
            print(f"Current tempo: {current_tempo}")

    input_stream = Input(filename=args.input if isinstance(args.input, str) else None,
                         device=args.input if isinstance(args.input, int) else None,
                         block_size=block_size, input_buffer=input_buffer, callback=btrack_callback)
    output_stream = Output(input_buffer, block_size=block_size, sample_rate=input_stream.sample_rate)
    input_sample_rate = input_stream.sample_rate
    print(f"Input sample rate: {input_sample_rate}")

    with open(args.loop, 'rb') as f:
        loop: AudioLoop = pickle.load(f)
    loop_output_stream = Output(loop_buffer, block_size=block_size, sample_rate=loop.sample_rate, device=args.output)

    stretcher = AudioStretcher(sample_rate=loop.sample_rate, channels=loop.channels, realtime=True)

    input_stream.start()
    output_stream.start()

    # start = False

    # def start_thread():
    #     input("Press enter to start loop playback")
    #     start = True
    # start_thread = Thread(target=start_thread)
    # start_thread.start()

    # while not start:
    #     time_scale = loop.temp / current_tempo
    #     stretched = stretcher.stretch(loop.get_block(0, block_size), time_scale)
    #     loop_idx = loop_buffer.put(0, stretched)

    input("Press enter to start loop playback")

    while processed_samples <= block_size:
        start = time.perf_counter()

        time_scale = loop.tempo / current_tempo

        stretcher.set_time_ratio(time_scale)
        stretcher.process(loop.get_next_block(block_size), False)

        stretched = stretcher.retrieve()
        while stretched.shape[0] > 0:
            loop_idx = loop_buffer.put(loop_idx, stretched)
            processed_samples += np.shape(stretched)[0]

            stretched = stretcher.retrieve()

        print(f"Loop time: {time.perf_counter() - start}")

    loop_output_stream.start()

    while True:
        start = time.perf_counter()

        time_scale = loop.tempo / current_tempo

        stretcher.set_time_ratio(time_scale)
        stretcher.process(loop.get_next_block(block_size), False)

        stretched = stretcher.retrieve()
        while stretched.shape[0] > 0:
            processed_samples += np.shape(stretched)[0]
            next_idx = loop_buffer.get_next_idx(loop_idx, np.shape(stretched)[0])
            while (processed_samples - loop_output_stream.processed_samples) >= buf_size:
                time.sleep(0.005)
            loop_idx = loop_buffer.put(loop_idx, stretched)
            stretched = stretcher.retrieve()

        print(f"Loop time: {time.perf_counter() - start}")


if __name__ == "__main__":
    main()
