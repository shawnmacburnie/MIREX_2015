__author__ = 'shawn'

from scipy.io.wavfile import read
import matplotlib.pyplot as plt
def show_amplitude(file_name):
    input_data = read(file_name)
    audio = input_data[1]
    # plot the first 1024 samples
    plt.plot(audio)
    plot_sampledBeats(input_data[0], file_name, max(audio))
    # label the axes
    plt.ylabel("Amplitude")
    plt.xlabel("Time (samples)")
    # set the title
    plt.title(file_name)
    # display the plot
    plt.show()

import scipy
from scipy.signal import hann
from scipy.fftpack import rfft
def show_magnitued(file_name):
    # read audio samples
    input_data = read(file_name)
    audio = input_data[1]
    print(audio)
    # apply a Hanning window
    window = hann(1024)
    audio = audio[0:1024] * window
    # fft
    mags = abs(rfft(audio))
    # convert to dB
    mags = 20 * scipy.log10(mags)
    # normalise to 0 dB max
    # mags -= max(mags)
    file = open('tmp.txt', 'w')
    for i in mags:
        file.write(str(i) + '\n')
    file.close()
    # plot
    plt.plot(mags)
    # label the axes
    plt.ylabel("Magnitude (dB)")
    plt.xlabel("Frequency Bin")
    # set the title
    plt.title(file_name + " Spectrum")
    plt.show()

def plot_sampledBeats(sample_rate, file_name, maxValue):
    lines = [line.rstrip('\n') for line in open(file_name.replace('.wav', '.txt'))]
    maxValue = 0
    index = 0
    for line in lines:
        domain = []
        points = line.split('\t')
        range = [maxValue + index] * len(points)
        index += 1000
        for point in points:
            domain += [float(point) * sample_rate]
        plt.plot(domain,range, 'ko')

def seconds_to_sample(seconds, sampleRate):
    return seconds * sampleRate


show_amplitude("train5.wav")
# show_magnitued("train1.wav")
