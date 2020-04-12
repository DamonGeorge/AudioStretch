import numpy as np


class CircularBuffer(object):

    def __init__(self, shape: tuple):
        self.buffer = np.zeros(shape)
        self.buf_size = np.shape(self.buffer)[0]

    def get_next_idx(self, start_idx, length) -> int:
        return (start_idx + length) % self.buf_size

    def put(self, idx, data: np.ndarray, length=None) -> int:
         # handle wrap around of the buffer
        if length is None:
            length = np.shape(data)[0]

        if length > self.buf_size:
            raise ValueError("given length is larger than circular buffer")

        if idx + length >= self.buf_size:
            frames_left = self.buf_size - idx
            extra_frames = length - frames_left
            self.buffer[idx:] = data[:frames_left]
            self.buffer[:extra_frames] = data[frames_left:length]
            return extra_frames
        else:
            self.buffer[idx: idx + length] = data[:length]
            return idx + length

    def get_into(self, idx, output: np.ndarray, length=None) -> int:
        if length is None:
            length = np.shape(output)[0]

        if length > self.buf_size:
            raise ValueError("given length is larger than circular buffer")

        if idx + length >= self.buf_size:
            frames_left = self.buf_size - idx
            extra_frames = length - frames_left
            output[:frames_left] = self.buffer[idx:]
            output[frames_left:length] = self.buffer[:extra_frames]
            return extra_frames
        else:
            output[:length] = self.buffer[idx: idx + length]
            return idx + length

    def get(self, idx, length):
        output = np.zeros((length, *np.shape(self.buffer)[1:]))

        next_idx = self.get_into(idx, output)

        return next_idx, output
