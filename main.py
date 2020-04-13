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
from threading import Thread, Event


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
    # parse the command line arguments
    args = parse_args()
    # block size and buffer sizes
    block_size = args.block_size
    buf_size = block_size*20
    # create the io buffers
    input_buffer = CircularBuffer((buf_size, 2))
    loop_buffer = CircularBuffer((buf_size, 2))
    loop_idx = 0
    processed_samples = 0
    input_sample_rate = 44100

    # the beat tracker object
    btrack = BeatTracker(hop_size=512, frame_size=block_size)
    btrack.fix_tempo(120.0)
    current_tempo = 120
    beat_event = Event()

    # the input btrack callback
    def btrack_callback(block: np.ndarray):
        nonlocal btrack, current_tempo, input_sample_rate, beat_event

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

            beat_event.set()

    # the io stream for the streaming input
    input_stream = Input(filename=args.input if isinstance(args.input, str) else None,
                         device=args.input if isinstance(args.input, int) else None,
                         block_size=block_size, input_buffer=input_buffer, callback=btrack_callback)
    output_stream = Output(input_buffer, block_size=block_size, sample_rate=input_stream.sample_rate, gain=0.6)
    input_sample_rate = input_stream.sample_rate
    print(f"Input sample rate: {input_sample_rate}")

    # load the Audio Loop object and create its output stream
    loop = AudioLoop.from_file(args.loop)
    btrack.fix_tempo(loop.tempo)
    # loop_output_event = Event()

    # def loop_callback(*args):
    #     loop_output_event.set()

    loop_output_stream = Output(loop_buffer, block_size=block_size,
                                sample_rate=loop.sample_rate, device=args.output)  # , callback=loop_callback)

    # the time stretcher object
    stretcher = AudioStretcher(sample_rate=loop.sample_rate, channels=loop.channels, realtime=True)

    # start the io streams
    input_stream.start()
    output_stream.start()

    # wait to start loop playback until the user says so
    input("Press enter to start loop playback")

    # process a block of samples before starting loop playback
    while processed_samples < block_size:
        start = time.perf_counter()  # just for debugging

        # calc the time scale
        time_scale = loop.tempo / current_tempo

        # stretch the audio
        stretcher.set_time_ratio(time_scale)
        stretcher.process(loop.get_next_block(block_size), False)

        # retrieve stretched audio in a loop until no more audio available
        stretched = stretcher.retrieve()
        while stretched.shape[0] > 0:
            loop_idx = loop_buffer.put(loop_idx, stretched)
            processed_samples += np.shape(stretched)[0]

            stretched = stretcher.retrieve()

        # print(f"Loop time: {time.perf_counter() - start}")

    # start the loop output
    loop_output_stream.start()

    beat_event.clear()
    # loop_output_event.set()

    # the main processing loop
    while True:
        # loop_output_event.wait()
        # loop_output_event.clear()
        while (processed_samples - loop_output_stream.processed_samples) >= block_size:
            # print("first sleep")
            time.sleep(0.005)

        start = time.perf_counter()  # just for debugging

        if beat_event.is_set():  # we just had beat
            beat_event.clear()
            samples_til_next_input_beat = input_sample_rate * 60 // current_tempo
            samples_til_next_loop_beat = loop.get_samples_til_next_beat()
            print(f"loop beat idx = {loop.beat_idx}")
            print(f"samples till next input beat = {samples_til_next_input_beat}")
            print(f"samples till next loop beat = {samples_til_next_loop_beat}")

            if samples_til_next_loop_beat > samples_til_next_input_beat:  # this is RARE
                # we must compress/speed up the loop
                time_scale = samples_til_next_input_beat / samples_til_next_loop_beat
                print("First scale IF")

            # else if loop is behind the coming beat, we need to stretch/slow the loop
            elif samples_til_next_loop_beat > 0.5 * samples_til_next_input_beat \
                    and (samples_til_next_input_beat - samples_til_next_loop_beat) >= block_size:
                time_scale = samples_til_next_input_beat / samples_til_next_loop_beat
                print("Second scale IF")

            # else if loop if slightly ahead, we need to compress/speed up the loop
            elif samples_til_next_loop_beat < 0.5 * samples_til_next_input_beat \
                    and samples_til_next_loop_beat >= block_size:
                time_scale = samples_til_next_input_beat / \
                    (samples_til_next_loop_beat + loop.get_sample_length_of_next_beat())
                print("Third scale IF")
            else:
                # calc the time scale using tempos
                time_scale = loop.tempo / current_tempo
                print("Last scale IF")

            print(f"time scale = {time_scale}")

        # stretch the audio
        stretcher.set_time_ratio(time_scale)
        stretcher.process(loop.get_next_block(block_size), False)

        # retrieve stretched audio in a loop until no more audio available
        stretched = stretcher.retrieve()
        while stretched.shape[0] > 0:
            processed_samples += np.shape(stretched)[0]
            next_idx = loop_buffer.get_next_idx(loop_idx, np.shape(stretched)[0])
            # wait until there is room in the circular loop outpu buffer
            while (processed_samples - loop_output_stream.processed_samples) >= buf_size:
                print("sleep")
                time.sleep(0.005)
            loop_idx = loop_buffer.put(loop_idx, stretched)
            stretched = stretcher.retrieve()

        # print(f"Loop time: {time.perf_counter() - start}")


if __name__ == "__main__":
    main()
