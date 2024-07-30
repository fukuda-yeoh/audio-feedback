import os

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv
import numpy as np

from audio_feedback.camera import CameraThread, load_intrinsic, load_extrinsic
from audio_feedback.defs import project_root
from audio_feedback.recognition import HSVColorModel, RecognitionThread
from audio_feedback.tones import Listener, Sound, Source

camera_no = 0

intrinsic_matrix, distortion_coeffs, fisheye = load_intrinsic(
    project_root() / "calibration" / "intrinsic.json"
)  # load undistort calibration

# モデルのHSVの設定
model = HSVColorModel(
    hue_range=(100, 120), saturation_range=(200, 255), value_range=(110, 255)
)

# setup camera thread
camera_thread = CameraThread(camera_no)
if "intrinsic_matrix" in locals():
    camera_thread.set_undistort(intrinsic_matrix, distortion_coeffs, fisheye)

# setup recognition thread
recognition_thread = RecognitionThread(model)

# 音のフィードバックの設定
listener = Listener()
source = Source()

# initialise sound
sound_file = project_root() / "sound_files" / "droplet.wav"
my_sound = Sound(sound_file)

# set listener and source positions (初期位置)
listener.position = (0, 240, 0)  # 画面左端の中央
listener.orientation = ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # x方向(右方向)，z方向（上方向）を向いている

# load sound into source
source.add_sound(my_sound)
source.loop = True
source.rolloff = 0.03 #音量の減衰の仕方を決める
source.play()
source_sound_on = True

def calculate_volume(distance):
    # 最大音量と最小音量の範囲を設定
    max_volume = 10.0
    min_volume = 1.0
    
    # 距離に応じて音量を計算
    volume = max_volume - (distance / 5.0) * (max_volume - min_volume)
    
    # 音量の範囲を0.0~1.0に収める
    return max(min(volume, max_volume), min_volume)


# run
camera_thread.start()
recognition_thread.start()

while True:
    import time
    frame = camera_thread.queue.get()
    recognition_thread.in_queue.put(frame)
    result = recognition_thread.out_queue.get()

    if result:
        center = result.center
        img_width, img_height, _ = frame.shape
        
        # ボールの位置
        ball_position = (center[0], center[1], 0)
        source.position = ball_position


        # プレイヤーとの距離を計算
        distance = np.linalg.norm(np.array(ball_position[:2]) - np.array(listener.position[:2]))
        volume = calculate_volume(distance)
        
        # 音量を設定
        source.volume = volume

        if not source_sound_on:
            source.play()
            source_sound_on = True
    else:
        source.stop()
        source_sound_on = False

    cv.imshow(f"Camera {camera_no}", result.annotated_img)

    if cv.waitKey(1) == ord("q"):
        recognition_thread.stop()
        recognition_thread.in_queue.put(frame)
        camera_thread.stop()
        break

cv.destroyAllWindows()