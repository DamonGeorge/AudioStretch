
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
import librosa

# input


# output
class Loop(object):
    def __init__(self, path="/Users/damongeorge/Music/Canteloupe Island.wav", estimated_bpm=120.0, hop_length=512, block_size=1024):
        file = sf.SoundFile(path)

        self.audio, self.sample_rate = file.read(dtype="float32")

        mono = librosa.to_mono(self.audio)

        self.tempo, self.beats = librosa.beat.beat_track(mono.T, sr=self.sample_rate,
                                                         hop_length=hop_length, start_bpm=estimated_bpm, units='samples')
        self.block_size = 1024
        self.samples = self.audio.shape[0]
        self.channels = self.audio.shape[1]
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
