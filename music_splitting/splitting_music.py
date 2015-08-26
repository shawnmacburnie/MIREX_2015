#!/usr/bin/python
import sys
"""
USAGE:

basically have all music file you want to be split in the folder.

Then in terminal enter the commant  'ls | ./splitting_music.py'
"""
from scipy.io.wavfile import read
from scipy.io.wavfile import write

def split_file(file_name):
    print("splitting up " + file_name)
    input_data = read(file_name)
    s_rate = input_data[0]
    audio = input_data[1]

    start_time = 0
    jump_size = 5 * s_rate
    file_counter = 1
    while start_time + jump_size < len(audio):
        write(file_name.replace('.wav','') + 'Split' + str(file_counter) + '.wav', s_rate,audio[start_time:start_time + jump_size])
        file_counter += 1
        start_time += jump_size

data = sys.stdin.read()
data = data.split('\n')
print("starting ... ")
if len(data):
    for i in data:
        if not '.py' in i and '.wav' in i:
            split_file(i)


print("finished")