import numpy as np
from threading import Event
from circular_buffer import CircularBuffer


class QueueBuffer(object):
    """
    Queue using Circular Buffer Implementation

    Allows AT MOST 1 reader and 1 writer threads -> NO MULTI READERS OR WRITERS
    """

    def __init__(self, shape: tuple = (0, 0), buffer=None):
        self.buffer = CircularBuffer(shape, buffer=buffer)
        assert self.capacity % 2 == 0, "buffer size must be divisible by 2"

        self.read_idx = 0
        self.write_idx = 0
        self.read_event = Event()
        self.write_event = Event()

    @property
    def capacity(self):
        return self.buffer.buf_size

    def empty(self):
        return self.read_idx == self.write_idx

    def size(self):
        return self.write_idx - self.read_idx

    def full(self):
        return self.size() == self.capacity

    def put(self, data: np.ndarray, length=None, put_incrementally=False) -> int:
        if length is None:
            length = np.shape(data)[0]

        if not put_incrementally:
            self.read_event.clear()
            while self.write_idx + length - self.read_idx > self.capacity:  # we must wait
                self.read_event.wait()
                self.read_event.clear()

            self.buffer.put(self.write_idx % self.capacity, data, length=length)
            self.write_idx += length
            self.write_event.set()
        else:
            self.read_event.clear()
            if self.write_idx + length - self.read_idx <= self.capacity:  # we can put everything in
                self.buffer.put(self.write_idx % self.capacity, data, length=length)
                self.write_idx += length
                self.write_event.set()
            else:
                remaining = length

                while remaining > 0:
                    # get available space
                    avail = self.capacity - self.size()
                    # clear event
                    self.read_event.clear()
                    # wait for space and loop again if necessary
                    if avail == 0:
                        self.read_event.wait()
                        continue

                    # don't add to much
                    if avail > remaining:
                        avail = remaining

                    # fill the available space
                    self.buffer.put(self.write_idx % self.capacity, data[-remaining:], length=avail)
                    # update write_idx and write_event
                    self.write_idx += avail
                    self.write_event.set()
                    # update remaining
                    remaining -= avail
                    # wait if necessary:
                    if remaining > 0:
                        self.read_event.wait()

        return True

    def put_nowait(self, data: np.ndarray, length=None):
        if length is None:
            length = np.shape(data)[0]

        if self.write_idx + length - self.read_idx > self.capacity:
            return False
        else:
            self.buffer.put(self.write_idx % self.capacity, data, length=length)
            self.write_idx += length
            self.write_event.set()
            return True

    # def put_force(self, data: np.ndarray, length=None):
    #     if length is None:
    #         length = np.shape(data)[0]

    #     # put no matter what
    #     self.buffer.put(self.write_idx % self.capacity, data, length=length)
    #     self.write_idx += length

    #     if self.write_idx - self.read_idx > self.capacity:  # bump up read idx
    #         self.read_idx += length

    #     self.write_event.set()
    #     return True

    def get_into(self, output: np.ndarray, length=None) -> int:
        if length is None:
            length = np.shape(output)[0]

        self.read_event.clear()
        while self.read_idx + length > self.write_idx:
            self.read_event.wait()
            self.read_event.clear()

        self.buffer.get_into(self.read_idx % self.capacity, output, length=length)
        self.read_idx += length
        self.read_event.set()
        return True

    def get_into_nowait(self, output: np.ndarray, length=None) -> int:
        if length is None:
            length = np.shape(output)[0]

        if self.read_idx + length > self.write_idx:
            return False
        else:
            self.buffer.get_into(self.read_idx % self.capacity, output, length=length)
            self.read_idx += length
            self.read_event.set()
            return True

    def get(self, length):
        self.read_event.clear()
        while self.read_idx + length > self.write_idx:
            self.read_event.wait()
            self.read_event.clear()

        output = self.buffer.get(self.read_idx % self.capacity, length)
        self.read_idx += length
        self.read_event.set()
        return output

    def get_nowait(self, length):
        if self.read_idx + length > self.write_idx:
            return None
        else:
            output = self.buffer.get(self.read_idx % self.capacity, length)
            self.read_idx += length
            self.read_event.set()
            return output

    # def get_into_force(self, output: np.ndarray, length=None) -> int:
    #     if length is None:
    #         length = np.shape(output)[0]

    #     # get no matter what
    #     self.buffer.get_into(self.read_idx % self.capacity, output, length=length)
    #     self.read_idx += length

    #     if self.read_idx > self.write_idx:  # let's force it -> bump up write idx
    #         self.write_idx = self.read_idx

    #     self.read_event.set()
    #     return True

    # def get_force(self, length):
    #     # get no matter what
    #     output = self.buffer.get(self.read_idx % self.capacity, length)
    #     self.read_idx += length

    #     if self.read_idx > self.write_idx:  # let's force it -> bump up write idx
    #         self.write_idx = self.read_idx

    #     self.read_event.set()
    #     return output
