import numpy as np
from scipy.io.wavfile import write
from scipy import signal


def gen_sine(
    freq, # 生成する正弦波の周波数（Hz）
    amplitude=np.iinfo(np.int16).max, # 波の振幅（デフォルトは int16 の最大値）。
    duration=1.0, # 波の持続時間（秒）。
    sample_rate=44100, # サンプリングレート（デフォルトは 44100 Hz）
    zero_padding_duration=0.0, # 波の後に追加する無音時間（秒）
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

    #断続音　
    # gen_sine(200, duration=0.5)
    # t, y, fs = gen_sine(1000, duration=0.05,zero_padding_duration=0.1)

    #連続音
    #t, y, fs = gen_sine(1000, duration=10)
    # plt.plot(t, y)
    # plt.show()

    # WAVファイルとして保存
    # path = project_root() / "sound_files" / "1000Hz_v3.wav"
    # write(path, fs, y.astype(np.int16))

    #repeat_sound
    num_repeats = 30  # 10回繰り返す
    t_all = []
    y_all = []

    for _ in range(num_repeats):
        t, y, fs = gen_sine(1000, duration=0.05, zero_padding_duration=0.1)
        t_all.append(t + (t_all[-1][-1] + 1/fs) if t_all else t)  # 時間軸をずらして連結
        y_all.append(y)

    # すべての波形を結合
    t_final = np.concatenate(t_all)
    y_final = np.concatenate(y_all)

    # WAVファイルとして保存
    path = project_root() / "sound_files" / "1000Hz_repeated.wav"
    write(path, fs, y_final.astype(np.int16))

   
