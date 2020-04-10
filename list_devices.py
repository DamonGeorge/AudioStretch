""" 
Run this simple script to print the list of 
available audio devices.
"""
import sounddevice as sd


print(sd.query_devices())
