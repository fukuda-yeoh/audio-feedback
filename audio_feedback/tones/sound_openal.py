import ctypes
import wave
from math import radians, cos, sin

from openal import al, alc

format_map = {
    (1, 8): al.AL_FORMAT_MONO8,
    (2, 8): al.AL_FORMAT_STEREO8,
    (1, 16): al.AL_FORMAT_MONO16,
    (2, 16): al.AL_FORMAT_STEREO16,
}


class Listener:
    def __init__(self):
        # load device/context/listener
        self.device = alc.alcOpenDevice(None)
        self.context = alc.alcCreateContext(self.device, None)
        alc.alcMakeContextCurrent(self.context)
        self._position = (0.0, 0.0, 0.0)
        self._velocity = (0.0, 0.0, 0.0) #速度
        self._orientation = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)) #方向

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos
        al.alListener3f(al.AL_POSITION, *pos)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel):
        self._velocity = vel
        al.alListener3f(al.AL_VELOCITY, *vel)

    @property
    def orientation(self):
        return self._orientation

    @orientation.setter
    def orientation(self, orientation):
        self._orientation = orientation
        al.alListenerfv(
            al.AL_ORIENTATION,
            (ctypes.c_float * 6)(*self._orientation[0], *self._orientation[1]),
        )

    def __del__(self):
        alc.alcDestroyContext(self.context)
        alc.alcCloseDevice(self.device)


class Source:
    # load default settings
    def __init__(self):
        # load source player
        self.source = al.ALuint(0)
        al.alGenSources(1, self.source)
        # disable rolloff factor by default
        al.alSourcef(self.source, al.AL_ROLLOFF_FACTOR, 0)
        # disable source relative by default
        al.alSourcei(self.source, al.AL_SOURCE_RELATIVE, 0)
        # capture player state buffer
        self.state = al.ALint(0)
        # set internal variable tracking
        self._volume = 1.0
        self._pitch = 1.0
        self._position = (0.0, 0.0, 0.0)
        self._velocity = (0.0, 0.0, 0.0) #速度
        self._rolloff = 1.0
        self._loop = False

        self.queue = []

    @property
    def rolloff(self):
        return self._rolloff

    @rolloff.setter
    def rolloff(self, value):
        self._rolloff = value
        al.alSourcef(self.source, al.AL_ROLLOFF_FACTOR, value)

    @property
    def loop(self):
        return self._loop

    @loop.setter
    def loop(self, loop):
        # set whether looping or not - true/false 1/0

        self._loop = loop
        al.alSourcei(self.source, al.AL_LOOPING, loop)

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos
        al.alSource3f(self.source, al.AL_POSITION, *pos)

    @property
    def velocity(self):
        return self._velocity

    @velocity.setter
    def velocity(self, vel):
        self._velocity = vel
        al.alSource3f(self.source, al.AL_VELOCITY, *vel)

    @property
    def pitch(self):
        return self._pitch

    @pitch.setter
    def pitch(self, pitch):
        self._pitch = pitch
        al.alSourcef(self.source, al.AL_PITCH, pitch)

    @property
    def volume(self):
        return self._volume

    @volume.setter
    def volume(self, vol):
        self._volume = vol
        al.alSourcef(self.source, al.AL_GAIN, vol)

    def add_sound(self, sound):
        # queue a sound buffer
        al.alSourceQueueBuffers(self.source, 1, sound.buf)  # self.buf
        self.queue.append(sound)

    def remove_sound(self):
        # remove a sound from the queue (detach & unqueue to properly remove)
        if len(self.queue) > 0:
            al.alSourceUnqueueBuffers(self.source, 1, self.queue[0].buf)  # self.buf
            al.alSourcei(self.source, al.AL_BUFFER, 0)
            self.queue.pop(0)

    def play(self):
        # play sound source
        al.alSourcePlay(self.source)

    def playing(self):
        al.alGetSourcei(self.source, al.AL_SOURCE_STATE, self.state)
        if self.state.value == al.AL_PLAYING:
            return True
        else:
            return False

    def stop(self):
        # stop playing sound
        al.alSourceStop(self.source)

    def rewind(self):
        # rewind player
        al.alSourceRewind(self.source)

    def pause(self):
        # pause player
        al.alSourcePause(self.source)

    @property
    def seek(self):
        # returns current buffer length position (IE: 21000), so divide by the buffers self.length
        # returns float 0.0-1.0
        al.alGetSourcei(self.source, al.AL_BYTE_OFFSET, self.state)
        return float(self.state.value) / float(self.queue[0].length)

    @seek.setter
    def seek(self, offset):
        # float 0.0-1.0
        al.alSourcei(self.source, al.AL_BYTE_OFFSET, int(self.queue[0].length * offset))

    def __del__(self):
        # delete sound source
        al.alDeleteSources(1, self.source)


class Sound:
    def __init__(self, filename):
        with wave.open(str(filename)) as f:

            channels = f.getnchannels()
            bit_rate = f.getsampwidth() * 8
            sample_rate = f.getframerate()
            wav_buffer = f.readframes(f.getnframes())

            al_format = format_map[(channels, bit_rate)]

        self.duration = (len(wav_buffer) / float(sample_rate)) / 2
        self.length = len(wav_buffer)
        self.buf = al.ALuint(0)
        al.alGenBuffers(1, self.buf)

        # allocate buffer space to: buffer, format, data, len(data), and samplerate
        al.alBufferData(self.buf, al_format, wav_buffer, len(wav_buffer), sample_rate)

    def __del__(self):
        al.alDeleteBuffers(1, self.buf)


if __name__ == "__main__":
    from audio_feedback.defs import project_root
    import time

    listener = Listener()
    source = Source()

    # initialise sound
    sound_file = project_root() / "sound_files" / "droplet.wav"
    my_sound = Sound(sound_file)

    # set listener positions
    listener.position = (320, 240, 0)
    listener.orientation = (
        (1.0, 0.0, 0.0),  # front vector
        (0.0, 0.0, 1.0),  # up vector
    )

    # set source positions
    source.position = (0, 240, 0)

    # load sound into source
    source.add_sound(my_sound)
    source.loop = True
    source.rolloff = 0.01
    source.play()

    # move sound from left to right
    print("moving left to right")
    for a in range(0, 640, 10):
        source.position = (a, 240, 0)
        time.sleep(0.1)

    # rotate listener
    print("rotating listener a single round")
    source.position = (0, 240, 0)
    listener.position = (320, 240, 0)
    for theta in range(0, 360, 5):
        x = cos(radians(theta))
        y = sin(radians(theta))
        listener.orientation = (
            (x, y, 0.0),  # front vector
            (0.0, 0.0, 1.0),  # up vector
        )
        time.sleep(0.1)
        
    # stop player
    source.stop()

    del listener
    del source
    del my_sound
