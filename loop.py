
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa
from circular_buffer import CircularBuffer

# input


# output
class AudioLoop(object):
    def __init__(self, path="drummer_120.wav", estimated_bpm=120.0, hop_length=512, block_size=1024, align_beats_to_start=True):
        self.audio, self.sample_rate = sf.read(str(path), dtype="float32")

        self.buffer = CircularBuffer(buffer=self.audio)
        self.buffer.put(0, self.audio)
        self.buf_idx = 0

        self.block_size = block_size
        self.hop_length = hop_length
        self.samples = self.audio.shape[0]
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

        self.beat_samples = self.beat_frames * hop_length

        self.cur_idx = 0

    def get_block(self, idx, num_frames) -> np.ndarray:
        i, output = self.buffer.get(idx, num_frames)
        return output

    def get_block_into(self, idx, num_frames, buffer: np.ndarray):
        self.buffer.get_into(idx, buffer, num_frames)

    def get_next_block(self, num_frames: int) -> np.ndarray:
        self.buf_idx, output = self.buffer.get(self.buf_idx, num_frames)
        return output

    def get_next_block_into(self, num_frames: int, buffer: np.ndarray):
        self.buf_idx = self.buffer.get_into(self.buf_idx, buffer, num_frames)
