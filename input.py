
"""Load an audio file into memory and play its contents.

NumPy and the soundfile module (https://PySoundFile.readthedocs.io/)
must be installed for this to work.

This example program loads the whole file into memory before starting
playback.
To play very long files, you should use play_long_file.py instead.

"""
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import utils
from typing import Callable
from threading import Thread, Event


class Input(object):
    def __init__(self, input_buffer: np.ndarray, filename=None, device=None, block_size=512, sample_rate=44100, callback: Callable = utils.empty_func):
        self.buffer = input_buffer
        self.buf_size = np.shape(self.buffer)[0]
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.in_idx = 0
        self.start_event = Event()
        self.callback = callback

        self._stop = False

        if filename:
            self.from_file = True
            self.file = sf.SoundFile(filename)
            self.sample_rate = self.file.samplerate
            self.thread = Thread(target=self._stream_file)
        else:
            self.from_file = False
            self.stream = sd.InputStream(callback=self._stream_callback,
                                         samplerate=self.sample_rate, blocksize=self.block_size, device=device)

    def start(self):
        if self.from_file:
            self.thread.start()
        else:
            self.stream.start()

        print("WAITING")
        self.start_event.wait()

    def stop(self):
        if self.from_file:
            self._stop = True
        else:
            self.stream.stop()

    def abort(self):
        if self.from_file:
            self._stop = True
        else:
            self.stream.abort()

    def _handle_new_block(self, block, num_frames):
        self.callback(block)

        # handle wrap around of the buffer
        if self.in_idx + num_frames >= self.buf_size:
            frames_left = self.buf_size - self.in_idx
            extra_frames = num_frames - frames_left
            self.buffer[self.in_idx:] = block[:frames_left]
            self.buffer[:extra_frames] = block[frames_left:]
            self.in_idx = extra_frames

        else:
            self.buffer[self.in_idx: self.in_idx + num_frames] = block
            self.in_idx += num_frames

            # Signal start event after two blocks
            if not self.start_event.is_set() and self.in_idx > self.block_size*2:
                self.start_event.set()

    def _stream_file(self):
        """THREAD: Imitates real time audio stream but from file"""
        time_per_loop = (self.block_size / self.sample_rate)
        extra_time_slept = 0
        while not self._stop:
            start = time.perf_counter()

            # process the audio
            block = self.file.read(frames=self.block_size)
            num_frames = np.shape(block)[0]
            self._handle_new_block(block, num_frames)

            # compute processing time
            end_compute = time.perf_counter()
            elapsed_compute = end_compute - start

            # sleep for remaining time at given sample rate
            sleep_time = time_per_loop - elapsed_compute - extra_time_slept
            time.sleep(sleep_time)

            # record any excess time spent sleeping to remove in the next loop
            end_sleep = time.perf_counter()
            elapsed_sleep = end_sleep - end_compute
            extra_time_slept = elapsed_sleep - sleep_time

    def _stream_callback(self, indata: np.ndarray, num_frames: int,
                         time, status: sd.CallbackFlags) -> None:
        self._handle_new_block(indata, num_frames)
