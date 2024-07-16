import numpy as np
from scipy.io.wavfile import write


def gen_sine(freq, amplitude=np.iinfo(np.int16).max, duration=1.0, sample_rate=44100):
    time = np.arange(0.0, duration, 1 / sample_rate)
    data = amplitude * np.sin(2.0 * np.pi * freq * time)
    return time, data, sample_rate


if __name__ == "__main__":
    from audio_feedback.defs import project_root

    # from matplotlib import pyplot as plt

    gen_sine(200, duration=0.5)

    t, y, fs = gen_sine(200, duration=5)
    # plt.plot(t, y)
    # plt.show()

    path = project_root() / "sound_files" / "200hz.wav"
    write(path, fs, y.astype(np.int16))
