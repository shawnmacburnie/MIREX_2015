__author__ = 'shawn'

from scipy.io.wavfile import read
from scipy.io.wavfile import write

def add_tick(audio_file_name, tick_times):
    input_data = read(audio_file_name)
    s_rate = input_data[0]
    audio = input_data[1]

    tick_audio = read('tick.wav')[1]
    for i in tick_times:
        i = float(i)
        start_index = i * s_rate
        for i in range(0,len(tick_audio)):
            if start_index >= len(audio):
                break
            audio[start_index] += tick_audio[i][0]
            start_index += 1
    write('withTick.wav', s_rate,audio)


lines = [line.rstrip('\n') for line in open('train1.txt','r')]
add_tick('train1.wav',lines[0].split('\t'))