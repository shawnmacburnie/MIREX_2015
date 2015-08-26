__author__ = 'shawn'

from scipy.io.wavfile import read
def breakdown_file(file_name,n_second):
    input_data = read(file_name)
    s_rate = input_data[0]
    audio = input_data[1]

    sample_size = n_second * s_rate
    start_index = 0
    data = []
    while start_index < len(audio) - sample_size:
        # print(audio[start_index:start_index + sample_size])
        data.append(audio[start_index:start_index + sample_size])
        start_index += 1

    print("Number Samples: %d" %len(data))
    print("Sample Size: %d" %len(data[0]))



breakdown_file('train1.wav', 5)

