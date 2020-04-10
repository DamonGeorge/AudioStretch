# AudioStretch
Some fun audio I/O, beat detection, and stretching in Python

## Installation
Using Python3 and either miniconda or pip, import python-sounddevice, py-soundfile, librosa and numpy.
Google can help you with these; none of these libraries are hard to find

## Libraries Used
python-sounddevice:  wraps port_audio for audio I/O
py-soundfile: wraps libsndfile for reading/writing audio files
librosa: audio utility that includes beat detection, time stretching, and a bunch of other features

## main.py
Given a filename, this routes input from a file to the default output. The file input simulates an input stream which is very useful.

## btrack_test.py
This adds beat tracking to main.py. 'Beat' is printed to the screen every time a beat is detected from the input

IMPORTANT: This uses the BTrack Library Python Wrapper from my Python-BTrack library. 
The compiled binary from building the python wrapper should be placed into the lib/ folder
in order to run this file!
