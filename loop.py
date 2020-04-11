
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa

# input


# output
class AudioLoop(object):
    def __init__(self, path="drummer_120.wav", estimated_bpm=120.0, hop_length=512, block_size=1024, align_beats_to_start=True):
        self.audio, self.sample_rate = sf.read(str(path), dtype="float32")

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

    def get_next_block(self, buffer: np.ndarray, num_frames: int):
        next_idx = self.cur_idx + num_frames

        if next_idx >= self.samples:
            frames_left = self.samples - self.cur_idx
            extra_frames = num_frames - frames_left
            buffer[:frames_left] = self.audio[self.cur_idx:]
            buffer[frames_left:] = self.audio[:extra_frames]
            self.cur_idx = extra_frames
        else:
            buffer[:] = self.audio[self.cur_idx: next_idx]
            self.out_idx = next_idx
