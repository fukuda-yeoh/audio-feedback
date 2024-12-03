"""Demonstrates AngularPannedSource by moving a sound in a circle."""
import math
import time

import synthizer

from audio_feedback.defs import project_root

# initialise sound
sound_file = project_root() / "sound_files" / "droplet.wav"

with synthizer.initialized():
    context = synthizer.Context()
    context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
    context.default_distance_model.value = synthizer.DistanceModel.LINEAR

    buffer = synthizer.Buffer.from_file(str(sound_file))
    generator = synthizer.BufferGenerator(context)
    generator.buffer.value = buffer
    generator.looping.value = True

    source = synthizer.Source3D(context)
    # source = synthizer.AngularPannedSource(context)
    source.add_generator(generator)
    source.play()

    distance = 200
    # There are 361 steps (degrees) because we need to complete the circle from degree 359 back to 360
    for step in range(0, 361):
        angle = math.radians(step)
        x, y = math.cos(angle) * distance, math.sin(angle) * distance
        source.position.value = (x, y, 0)
        source.play()
        print(source.position.value)
        time.sleep(0.1)

    # for angle in range(0, 360):
    #     print(angle)
    #     source.azimuth.value = angle % 360
    #     time.sleep(0.05)
