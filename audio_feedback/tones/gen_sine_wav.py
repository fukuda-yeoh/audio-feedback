import numpy as np
from scipy.io.wavfile import write
from scipy import signal


def gen_sine(
    freq,
    amplitude=np.iinfo(np.int16).max,
    duration=1.0,
    sample_rate=44100,
    zero_padding_duration=0.0,
    window=None,
):
    #sin波生成
    time = np.arange(0.0, duration, 1 / sample_rate)
    data = amplitude * np.sin(2.0 * np.pi * freq * time)
    #指数
    a=25
    window_v2=np.exp(-a*time)
    #窓関数
    data = data * window_v2

    #空白
    zero_pad_time = np.arange(
        duration, duration + zero_padding_duration, 1 / sample_rate
    )
    zero_pad_data = np.zeros(zero_pad_time.shape)

    time = np.concatenate([time, zero_pad_time])
    data = np.concatenate([data, zero_pad_data])

    return time, data, sample_rate
    
    


if __name__ == "__main__":
    from audio_feedback.defs import project_root

    # from matplotlib import pyplot as plt

    gen_sine(200, duration=0.5)

    t, y, fs = gen_sine(1000, duration=0.05,zero_padding_duration=0.1)
    #t, y, fs = gen_sine(1000, duration=10)
    # plt.plot(t, y)
    # plt.show()

    path = project_root() / "sound_files" / "1000Hz_v2.wav"
    write(path, fs, y.astype(np.int16))
