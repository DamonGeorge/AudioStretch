
"""Load an audio file into memory and play its contents.

NumPy and the soundfile module (https://PySoundFile.readthedocs.io/)
must be installed for this to work.

This example program loads the whole file into memory before starting
playback.
To play very long files, you should use play_long_file.py instead.

"""
import argparse

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    '-l', '--list-devices', action='store_true',
    help='show list of audio devices and exit')
args, remaining = parser.parse_known_args()
if args.list_devices:
    print(sd.query_devices())
    parser.exit(0)
parser = argparse.ArgumentParser(
    description=__doc__,
    formatter_class=argparse.RawDescriptionHelpFormatter,
    parents=[parser])
parser.add_argument(
    'filename', metavar='FILENAME',
    help='audio file to be played back')
parser.add_argument(
    '-d', '--device', type=int_or_str,
    help='output device (numeric ID or substring)')
args = parser.parse_args(remaining)


try:
    sample_rate = sd.default.samplerate
    block_size = 512
    out_idx = 0
    in_idx = 0
    buf_size = 4096
    data = np.zeros((buf_size, 2))

    print(f"Default sample rate: {sample_rate}")

    def stream_callback(outdata: np.ndarray, num_frames: int,
                        time, status: sd.CallbackFlags) -> None:
        global out_idx, data

        if out_idx + num_frames >= buf_size:
            frames_left = buf_size - out_idx
            extra_frames = num_frames - frames_left
            outdata[:frames_left] = data[out_idx:]
            outdata[frames_left:] = data[:extra_frames]
            out_idx = extra_frames
        else:
            outdata[:] = data[out_idx: out_idx + num_frames]
            out_idx += num_frames

    stream = sd.OutputStream(blocksize=block_size, callback=stream_callback)
    stream.start()

    # data, fs = sf.read(args.filename, dtype='float32')
    for block in sf.blocks(args.filename, blocksize=block_size):
        num_frames = block.shape[0]
        if in_idx + num_frames >= buf_size:
            frames_left = buf_size - in_idx
            extra_frames = num_frames - frames_left
            data[in_idx:] = block[:frames_left]
            data[:extra_frames] = block[frames_left:]
            in_idx = extra_frames
        else:
            data[in_idx: in_idx + num_frames] = block
            in_idx += num_frames

            time.sleep(num_frames/47000)

        # plp_envelope = librosa.beat.plp(y=None, sr=22050, onset_envelope=None, hop_length=512,
        #                                 win_length=384, tempo_min=30, tempo_max=300, prior=None)

        # time.sleep(num_frames/50000)

    # while True:
    #     time.sleep(0.5)
    #     print(f"in_idx: {in_idx}, out_idx: {out_idx}")
    # sd.play(data, fs, device=args.device)
    # status = sd.wait()
except KeyboardInterrupt:
    stream.stop()
    parser.exit('\nInterrupted by user')

except Exception as e:
    parser.exit(type(e).__name__ + ': ' + str(e))
# if status:
#     parser.exit('Error during playback: ' + str(status))
