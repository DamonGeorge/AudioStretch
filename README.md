# AudioStretch
Some fun audio I/O, beat detection, and stretching in Python

## Python Libraries Used
- python-sounddevice:  wraps port_audio for audio I/O
- py-soundfile: wraps libsndfile for reading/writing audio files
- librosa: audio utility that includes beat detection, time stretching, and a bunch of other features

## C/C++ Libraries Used
- rubberband: advanced audio stretching. See my Python-Rubberband repo for a Python wrapper
- btrack: real time beat tracking. See my Python-BTrack repo for a Python wrapper

## Installation
Using Python3 and either miniconda or pip, import python-sounddevice, py-soundfile, librosa, aubio and numpy.
Google can help you with these; none of these libraries are hard to find. BTrack and Rubberband Python wrappers are required as well. These can be built using my Python-BTrack and Python-Rubberband repos and the resulting binaries should be placed in lib/.

## Scripts
#### aubio_test.py
Given an input filename or device, this tests aubio beat tracking by printing Beats in real time with the audio output.

#### btrack_test.py
Given an input filename or device, this tests BTrack beat tracking by print 'Beat' to the screen in real time.

#### io_test.py
Simply testing routing audio from input to output. This uses a QueueBuffer to route the audio data. This script is most useful for testing the InputFileStream, which attempts to imitate a real time audio stream using a file, which is difficult. 

#### list_devices.py
Simply lists all the audio I/O devices currently available.

#### parse_loop.py
Parses an audio file into the AudioLoop format. This detects the beats and tempo of the audio, and plays the audio with metronome clicks placed at the detected beats. This let's the user decide whether the beat tracking works and whether to save the loop to a file (this saves as a .pkl file which we can use with AudioLoop.from_file()).

#### stretch_test.py
This tests the rubberband library by simply stretching the given input audio file in real time to the output.

### main.py
This is the full AudioStretch program. This takes an input stream/file and saved audio loop. It then streams the input to the output and plays the loop when the user presses 'Enter'. When streaming the audio loop, it attempts to sync the tempo and beats of the loop to the input stream in real time. Currently WIP

## Utilities
#### circular_buffer.py
This contains the CircularBuffer class that basically wraps a numpy array and provides simple indexing capabilities to use the numpy array as a circular buffer. This contains NO STATE -> indices are returned from all the functions.

#### input.py
This contains the Input class which attempts to consolidate streaming and file inputs into one object. THIS IS DEPRECATED but is still used in some scripts that haven't been updated yet.

#### input_file_stream.py
This contains the InputFileStream class, which imitates a sounddevice input stream using a provided audio file. 

#### loop.py
This contains the AudioLoop class, which is a wrapper around audio that has detected beats and tempo. This provides utitilities for retrieved the number of samples for given beats, and it also has save/load capabilities.

#### output.py
This contains the Output class, which is basically a wrapper around a sounddevice output stream that uses a circular buffer. THIS IS DEPRECATED, but is still used in some scripts that haven't been updated yet. 

#### queue_buffer.py
This contains the QueueBuffer class, which implements a Queue that wraps a CircularBuffer. This allows at most 1 reader and 1 writer to use this Queue from separate threads. Multiple readers or writers is not supported in a multi-threaded environment. This basically just adds state and some events to the CircularBuffer class. 

This should now be used in place of the CircularBuffer in most cases!!!

#### utils.py
A couple utilities, such as an empty function uses as the default for callable arguments.

## Resources
#### drummer_120.wav
A simple drummer loop from LogicProX at 120 bpm.

#### click_120.wav
A simple click track generated using librosa at 120 bpm.

#### *_LOOP.pkl
Saved Audio Loops
