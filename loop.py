
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa
from circular_buffer import CircularBuffer
import pickle
from pathlib import Path
# input


# output
class AudioLoop(object):
    def __init__(self, path="drummer_120.wav", estimated_bpm=120.0, hop_length=512, block_size=1024, align_beats_to_start=True, data=None):
        if isinstance(data, dict):
             # just update by attribute name
            for key, val in data.items():
                self.__dict__[key] = val
        else:
            # These are all saved variables
            self.audio, self.sample_rate = sf.read(str(path), dtype="float32")

            self.block_size = block_size
            self.hop_length = hop_length
            self.samples = self.audio.shape[0]
            if len(self.audio.shape) == 1:
                self.audio = np.expand_dims(self.audio, axis=1)
            self.channels = self.audio.shape[1]

            self.tempo, self.beat_frames = librosa.beat.beat_track(librosa.to_mono(self.audio.T), sr=self.sample_rate,
                                                                   hop_length=self.hop_length, start_bpm=estimated_bpm, units='frames', trim=False)

            self.num_frames_adjusted = 0
            if align_beats_to_start:
                self.num_frames_adjusted = self.beat_frames[0]
                if self.num_frames_adjusted > 0:
                    print(f"Adjusted detected beats by {self.num_frames_adjusted * self.hop_length} samples")

                    if (self.num_frames_adjusted * self.hop_length / self.sample_rate) > 0.05:
                        print("Warning: aligning beats to the loop start adjusted beats by more than 50 ms!")

                    self.beat_frames = self.beat_frames - self.num_frames_adjusted

        # initialize unsaved variables
        self._init_unsaved_variables()

    def _init_unsaved_variables(self):
        # These are not saved
        self.buffer = CircularBuffer(buffer=self.audio)
        self.buf_idx = 0

        self.beat_samples = self.beat_frames * self.hop_length
        self.beat_idx = 0

        self.cur_idx = 0

    def save(self, filename):
        with open(filename, "wb") as f:
            data = {
                "audio": self.audio,
                "sample_rate": self.sample_rate,
                "beat_frames": self.beat_frames,
                "tempo": self.tempo,
                "block_size": self.block_size,
                "hop_length": self.hop_length,
                "samples": self.samples,
                "channels": self.channels,
                "num_frames_adjusted": self.num_frames_adjusted
            }
            pickle.dump(data, f)

    @classmethod
    def from_file(cls, filename):
        if not Path(filename).exists():
            raise FileNotFoundError()

        with open(filename, "rb") as f:
            data = pickle.load(f)

        return cls(data=data)

    # def get_block(self, idx, num_frames) -> np.ndarray:
    #     i, output = self.buffer.get(idx, num_frames)
    #     return output

    # def get_block_into(self, idx, num_frames, buffer: np.ndarray):
    #     self.buffer.get_into(idx, buffer, num_frames)

    def get_next_block(self, num_frames: int) -> np.ndarray:
        self.buf_idx, output = self.buffer.get(self.buf_idx, num_frames)
        self._increment_beat_idx()
        return output

    def get_next_block_into(self, num_frames: int, buffer: np.ndarray):
        self.buf_idx = self.buffer.get_into(self.buf_idx, buffer, num_frames)
        self._increment_beat_idx()

    def _increment_beat_idx(self):
        # if we are on the last beat idx
        if self.beat_idx == self.beat_samples.shape[0] - 1:
            # if buf_idx is past the first beat marker and less than the last beat indx
            if self.buf_idx >= self.beat_samples[0] and self.buf_idx < self.beat_samples[-1]:
                self.beat_idx = 0
        else:
            next_beat_idx = self.beat_idx + 1
            # if buf_idx is past the next beat marker, increment beat idx
            if self.buf_idx >= self.beat_samples[self.beat_idx + 1]:
                self.beat_idx += 1

    def get_samples_til_next_beat(self) -> int:
        # if we are on the last beat idx
        if self.beat_idx == self.beat_samples.shape[0] - 1:
            # return samples from beginning till first beat + samples left in buffer
            return self.beat_samples[0] + self.buffer.buf_size - self.buf_idx
        else:
            # return sample idx of next beat - current buffer idx
            return self.beat_samples[self.beat_idx+1] - self.buf_idx

    def get_sample_length_of_beat(self, beat_idx) -> int:
        beat_idx = beat_idx % self.beat_samples.shape[0]

        # if we are on the last beat idx
        if beat_idx == self.beat_samples.shape[0] - 1:
            # return samples from beginning till first beat + samples left in buffer since current beat
            return self.beat_samples[0] + self.buffer.buf_size - self.beat_samples[-1]
        else:
            # return difference of consecutive beat samples
            return self.beat_samples[beat_idx+1] - self.beat_samples[beat_idx]

    def get_sample_length_of_next_beat(self) -> int:
        return self.get_sample_length_of_beat(self.beat_idx + 1)
