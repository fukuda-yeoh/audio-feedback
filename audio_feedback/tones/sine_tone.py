from pysinewave import SineWave


class SineTone:
    def __init__(self, pitch_per_second=960, decibels_per_second=1000):
        super().__init__()

        self.left_tone = SineWave(
            channels=2,
            channel_side="l",
            pitch_per_second=pitch_per_second,
            decibels_per_second=decibels_per_second,
        )
        self.right_tone = SineWave(
            channels=2,
            channel_side="r",
            pitch_per_second=pitch_per_second,
            decibels_per_second=decibels_per_second,
        )

        self._freq = 200  # Hz
        self._vol = 0
        self._lr_vol_range = 30
        self._lr_balance = 0

        self.change_freq(self._freq)
        self._set_volume(self._vol)

    def change_freq(self, freq=None):
        if freq is None:
            freq = self._freq
        self.left_tone.set_frequency(freq)
        self.right_tone.set_frequency(freq)

    def change_vol(self, vol=None):
        if vol is None:
            vol = self._vol
        self._set_volume(vol=vol)

    def change_lr_balance(self, lr_balance=None):
        if lr_balance is None:
            lr_balance = self._lr_balance
        self._set_volume(lr_balance=lr_balance)

    def _set_volume(self, vol=None, lr_balance=None):
        if vol is not None:
            self._vol = vol
        if lr_balance is not None:
            self._lr_balance = lr_balance

        left_vol = self._vol - self._lr_vol_range * (1 + self._lr_balance)
        right_vol = self._vol - self._lr_vol_range * (1 - self._lr_balance)

        self.left_tone.set_volume(left_vol)
        self.right_tone.set_volume(right_vol)

    def play(self):
        self.left_tone.play()
        self.right_tone.play()

    def stop(self):
        self.left_tone.stop()
        self.right_tone.stop()


if __name__ == "__main__":
    import time

    tone = SineTone()
    tone.play()
    tone.change_freq(200)

    for a in range(0, 100, 2):
        lr_balance = (a - 50) / 100
        tone.change_lr_balance(lr_balance)
        time.sleep(0.1)
