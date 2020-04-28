import argparse
import numpy as np
import sounddevice as sd
import soundfile as sf
from utils.audio_loop import AudioLoop
from lib.btrack import BeatTracker  # pylint: disable=import-error,no-name-in-module
import librosa
import pickle
import time
from utils.circular_buffer import CircularBuffer
from lib.rubberband import AudioStretcher  # pylint: disable=import-error,no-name-in-module
from threading import Thread, Event
from queue import Queue, Empty
from utils.queue_buffer import QueueBuffer
from utils.input_file_stream import InputFileStream
from aubio import tempo as Tempo  # pylint: disable=no-name-in-module


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
    parser.add_argument("-i", "--input", type=int_or_str, default=None,
                        help="either input device # or file to stream as input")
    parser.add_argument("-o", "--output", type=int, default=None, help="output device #")
    parser.add_argument("-b", "--block-size", type=int, default=1024, help="audio block size in frames")

    return parser.parse_args()


def main():
    # parse the command line arguments
    args = parse_args()
    # block size
    block_size = args.block_size
    hop_size = block_size // 2
    input_sample_rate = 44100

    # load the Audio Loop object
    loop = AudioLoop.from_file(args.loop)

    # create the io buffers
    input_buffer = QueueBuffer((4*block_size, 2))
    loop_buffer = QueueBuffer((int(1 * block_size), loop.channels))
    input_queue = Queue()

    # Stream callbacks
    def input_callback(indata, frames, *args, **kwargs):
        input_buffer.put(indata, frames)
        input_queue.put_nowait(indata)  # TODO: need to copy??

    def output_callback(outdata, frames, *args, **kwargs):
        if not input_buffer.get_into_nowait(outdata, length=frames):
            outdata[:] = 0

    def loop_output_callback(outdata, frames, *args, **kwargs):
        if not loop_buffer.get_into_nowait(outdata, length=frames):
            outdata[:] = 0

    # select either input stream or file
    if args.input is None or isinstance(args.input, int):
        input_stream = sd.InputStream(samplerate=input_sample_rate, blocksize=block_size,
                                      device=args.input, dtype="float32", callback=input_callback)
    elif isinstance(args.input, str):
        input_stream = InputFileStream(args.input, block_size=block_size, callback=input_callback)
        input_sample_rate = input_stream.sample_rate
    else:
        raise ValueError("Bad input argument")
    print(f"Input sample rate: {input_sample_rate}")

    # output stream for the input stream
    output_stream = sd.OutputStream(samplerate=input_sample_rate, blocksize=block_size, channels=input_stream.channels,
                                    device=args.output, callback=output_callback)

    # output stream for the stretched loop
    loop_output_stream = sd.OutputStream(samplerate=loop.sample_rate, blocksize=block_size,
                                         device=args.output, channels=loop.channels, callback=loop_output_callback)

    # the time stretcher object
    stretcher = AudioStretcher(sample_rate=loop.sample_rate, channels=loop.channels, realtime=True)

    # the beat tracker object
    btrack = BeatTracker(hop_size=hop_size, frame_size=block_size)
    btrack.fix_tempo(loop.tempo)
    current_tempo = 120
    beat_event = Event()
    samples_since_last_input_beat = 0
    btrack_thread_alive = True

    # the aubio object
    aubio_tracker = Tempo(buf_size=block_size, hop_size=hop_size, samplerate=input_sample_rate)

    # the btrack thread
    def btrack_thread():
        nonlocal btrack, beat_event, samples_since_last_input_beat, btrack_thread_alive, current_tempo, hop_size

        while btrack_thread_alive:
            try:
                # get newest audio block from input queue
                block = input_queue.get(timeout=1)  # 1 second timeout
            except Empty:
                continue

            # to mono if necessary
            if block.ndim > 1:
                if block.shape[1] > 1:
                    block = librosa.to_mono(block.T)
                else:
                    block = np.squeeze(block)

            # process the audio with btrack
            btrack.process_audio(block)

            # process audio with aubio
            num_hops = block.shape[0] // hop_size
            for i in range(num_hops):
                aubio_tracker(block[i*hop_size:(i+1)*hop_size])

            if btrack.beat_due_in_current_frame():
                print("Beat")
                beat_event.set()
                # TODO: should this be set to size of current block or 0??
                samples_since_last_input_beat = np.shape(block)[0]
                # get tempos from our two trackers
                btrack_tempo = btrack.get_current_tempo_estimate() * (input_sample_rate / 44100)
                aubio_tempo = aubio_tracker.get_bpm()
                # average the tempos if aubio confidence high
                if aubio_tracker.get_confidence() > 0.0:
                    # adjust aubio tempo to be in same range as btrack
                    if aubio_tempo > 1.5 * btrack_tempo:
                        aubio_tempo /= 2
                    elif aubio_tempo < 0.5 * btrack_tempo:
                        aubio_tempo *= 2
                    tempo = (aubio_tempo + btrack_tempo) / 2
                else:
                    tempo = btrack_tempo

                if abs(tempo - current_tempo) > 0.1:
                    current_tempo = tempo
                    print(f"Current tempo: {current_tempo}")
            else:
                samples_since_last_input_beat += np.shape(block)[0]

    # start the thread
    btrack_thread = Thread(target=btrack_thread)
    btrack_thread.start()

    # start the io streams
    input_stream.start()
    output_stream.start()

    try:
        # wait to start loop playback until the user says so
        input("Press enter to start loop playback")

        # flag for starting loop output after first iteration
        loop_output_started = False

        # other counters
        samples_til_next_input_beat = np.inf
        samples_since_time_scale_calculated = 0
        beat_count = 3

        # other flags
        reset_time_scale = False

        # WAIT TILL NEXT BEAT
        current_beat_length = (input_sample_rate * 60 // current_tempo)
        if samples_since_last_input_beat >= 0.3 * current_beat_length:
            beat_event.clear()
            while not beat_event.is_set():  # TODO: this could result in the loop starting slightly behind
                print('sleep')
                time.sleep(0.002)

        beat_event.clear()  # clear before we start our loop
        time_scale = loop.tempo / current_tempo  # initialize time scaling

        # the main processing loop
        while True:
            start = time.perf_counter()  # just for debugging

            if beat_event.is_set():
                beat_event.clear()

                # count beats
                beat_count = (beat_count + 1) % 2

                if beat_count >= 0:
                    # samples until the next beat of input stream normalized to sample rate of the loop
                    samples_til_next_input_beat = (input_sample_rate * 60 // current_tempo -
                                                   samples_since_last_input_beat) * loop.sample_rate / input_sample_rate
                    # samples until next beat of loop at original timescale
                    samples_til_next_loop_beat = loop.get_samples_til_next_beat(
                    ) + (loop_buffer.size() / time_scale)  # buffer was stretched so adjust by timescale

                    print(f"loop beat idx = {loop.beat_idx}")
                    # print(f"samples since last input beat = {samples_since_last_input_beat}")
                    print(f"samples till next input beat = {samples_til_next_input_beat}")
                    print(f"samples till next loop beat = {samples_til_next_loop_beat}")
                    print(f"samples till next loop beat stretched = {samples_til_next_loop_beat * time_scale}")

                    # ADJUSTMENTS
                    samples_til_next_input_beat -= block_size*2
                    # beat matching!!

                    # if loop is ahead, we must compress/speed up the loop -> time_scale < 1
                    if samples_til_next_loop_beat > samples_til_next_input_beat:
                        print("First scale IF")
                        ratio = samples_til_next_input_beat / samples_til_next_loop_beat
                        print(f"ratio = {ratio}")
                        time_scale = ratio

                    # else if loop is behind the coming beat, we need to stretch/slow the loop -> time_scale > 1
                    elif samples_til_next_loop_beat > 0.5 * samples_til_next_input_beat:  # \
                            # and (samples_til_next_input_beat - samples_til_next_loop_beat) >= block_size:
                        print("Second scale IF")
                        ratio = samples_til_next_input_beat / samples_til_next_loop_beat
                        print(f"ratio = {ratio}")
                        time_scale = ratio

                    # else if loop if slightly ahead, we need to compress/speed up the loop
                    elif samples_til_next_loop_beat < 0.5 * samples_til_next_input_beat:  # \
                            # and samples_til_next_loop_beat >= block_size:
                        print("Third scale IF")
                        ratio = samples_til_next_input_beat / \
                            (samples_til_next_loop_beat + (loop.get_sample_length_of_next_beat() / time_scale))
                        print(f"ratio = {ratio}")
                        time_scale = ratio
                    else:
                        # calc the time scale using tempos
                        time_scale = loop.tempo / current_tempo
                        print("Last scale IF")

                    # for testing
                    # time_scale = loop.tempo / current_tempo

                    print(f"time_scale = {time_scale}")
                    reset_time_scale = True

            elif reset_time_scale and samples_since_last_input_beat >= (input_sample_rate * 60 // current_tempo):
                time_scale = loop.tempo / current_tempo
                print(f"resetting time_scale = {time_scale}")
                reset_time_scale = False

            # stretch the audio
            stretcher.set_time_ratio(time_scale)
            stretcher.process(loop.get_next_block(block_size), False)

            # increment counter
            samples_since_time_scale_calculated += block_size

            # retrieve stretched audio in a loop until no more audio available
            stretched = stretcher.retrieve()
            while stretched.shape[0] > 0:
                # wait to put new samples into loop output buffer
                loop_buffer.put(stretched, put_incrementally=True)

                # see if we have more to retrieve
                stretched = stretcher.retrieve()

            # start the loop output stream if this was the first loop iteratoin
            if not loop_output_started:
                loop_output_started = True
                loop_output_stream.start()
                print("loop output started")

            # print(f"Loop time: {time.perf_counter() - start}")
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)
    finally:
        input_stream.stop()
        output_stream.stop()
        loop_output_stream.stop()
        btrack_thread_alive = False


if __name__ == "__main__":
    main()
