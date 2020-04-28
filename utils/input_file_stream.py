
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
import utils.helpers as utils
from typing import Callable, Optional
from threading import Thread, Event
from utils.circular_buffer import CircularBuffer


# TODO: handle wrapping of file
class InputFileStream(object):
    def __init__(self, filename, block_size=512, callback: Callable = utils.empty_func):
        self.filename = filename
        self.block_size = block_size
        self.callback = callback

        self.file = sf.SoundFile(filename)
        self.sample_rate = self.file.samplerate
        self.channels = self.file.channels

        self.start_event = Event()

        self.blocks_received = 0
        self._stop = False

    def start(self):
        self._stop = False
        self.start_event.clear()
        self.thread = Thread(target=self._stream_file)
        self.thread.start()

        print("WAITING")
        self.start_event.wait()

    def stop(self):
        self._stop = True

    def abort(self):
        self._stop = True

    def _stream_file(self):
        """THREAD: Imitates real time audio stream but from file"""
        time_per_loop = (self.block_size / self.sample_rate)
        extra_time_slept = 0
        while not self._stop:
            start = time.perf_counter()

            # process the audio
            block = self.file.read(frames=self.block_size, dtype='float32')
            if block.ndim == 1:
                block = np.expand_dims(block, axis=1)
            num_frames = np.shape(block)[0]
            self.blocks_received += num_frames
            self.callback(block, num_frames)

            # compute processing time
            end_compute = time.perf_counter()
            elapsed_compute = end_compute - start
            # print(f"input process time: {elapsed_compute}")

            # sleep for remaining time at given sample rate
            sleep_time = time_per_loop - elapsed_compute - extra_time_slept
            # print(f"Sleep time: {sleep_time}")
            time.sleep(max(0, sleep_time))

            # signal first block done
            if self.blocks_received <= self.block_size and self.blocks_received > 0:
                self.start_event.set()

            # record any excess time spent sleeping to remove in the next loop
            end_sleep = time.perf_counter()
            elapsed_sleep = end_sleep - end_compute
            extra_time_slept = elapsed_sleep - sleep_time
