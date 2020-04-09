
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


# input


# output
class Output(object):
    def __init__(self, output_buffer: np.ndarray, device=None, block_size=512, sample_rate=44100):
        self.buffer = output_buffer
        self.buf_size = np.shape(self.buffer)[0]
        self.block_size = block_size
        self.sample_rate = sample_rate
        self.out_idx = 0

        self.stream = sd.OutputStream(callback=self._stream_callback, blocksize=self.block_size,
                                      samplerate=self.sample_rate, device=device)

    def start(self):
        self.stream.start()

    def stop(self):
        self.stream.stop()

    def abort(self):
        self.stream.abort()

    def _stream_callback(self, outdata: np.ndarray, num_frames: int,
                         time, status: sd.CallbackFlags) -> None:
        if self.out_idx + num_frames >= self.buf_size:
            frames_left = self.buf_size - self.out_idx
            extra_frames = num_frames - frames_left
            outdata[:frames_left] = self.buffer[self.out_idx:]
            outdata[frames_left:] = self.buffer[:extra_frames]
            self.out_idx = extra_frames
            # print(f"Cleared output buffer!   out_idx={self.out_idx}")
        else:
            outdata[:] = self.buffer[self.out_idx: self.out_idx + num_frames]
            self.out_idx += num_frames
