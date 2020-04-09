
"""
This program 
"""
import argparse

import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import librosa
import asyncio
from input import Input
from output import Output


def int_or_str(text):
    """Helper function for argument parsing."""
    try:
        return int(text)
    except ValueError:
        return text


# Parse the command line args
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


async def main():
    sample_rate = 44100  # sd.default.samplerate
    block_size = 1024
    out_idx = 0
    in_idx = 0
    buf_size = 10240
    data = np.zeros((buf_size, 2))

    print(f"Default sample rate: {sample_rate}")
    input = Input(data, filename=args.filename, block_size=block_size)
    output = Output(data, block_size=block_size)
    try:
        print(f"Input Starting: {time.perf_counter()}")
        await input.start()
        print(f"Output Starting: {time.perf_counter()}")
        output.start()

        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        print('\nInterrupted by user')

    except Exception as e:
        print(type(e).__name__ + ': ' + str(e))

    finally:
        output.stop()
        input.stop()


# run main in the event loop
asyncio.run(main())
