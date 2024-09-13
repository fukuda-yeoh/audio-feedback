import os

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv
import numpy as np
import time

from audio_feedback.camera import CameraThread, load_intrinsic, load_extrinsic
from audio_feedback.defs import project_root
from audio_feedback.recognition import HSVColorModel, RecognitionThread
from audio_feedback.tones import Listener, Sound, Source


p_id = 1
condition = 1

camera_no = 1

intrinsic_matrix, distortion_coeffs, fisheye = load_intrinsic(
    project_root() / "calibration" / "test3.json"
)  # load undistort calibration

# モデルのHSVの設定
model = HSVColorModel(
    hue_range=(100, 130), saturation_range=(180, 240), value_range=(110, 255)
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
sound_file = project_root() / "sound_files" / "5000Hz.wav"
my_sound = Sound(sound_file)

# set listener and source positions (初期位置)
listener.position = (0, 240, 0)  # 画面左端の中央
listener.orientation = ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # x方向(右方向)，z方向（上方向）を向いている

# load sound into source
source.add_sound(my_sound)
source.loop = True
source.rolloff = 3 #音量の減衰の仕方を決める
source.play()
source_sound_on = True

def calculate_volume(distance, reference_distance=1.0, max_volume=200.0, min_volume=20.0):
    # 距離が0に近づきすぎると無限大になってしまうのを防ぐために、距離に下限を設けます
    if distance < reference_distance:
        distance = reference_distance
    
    # 音量を計算する
    volume = max_volume / (distance / reference_distance) ** 2
    
    # 音量をmax_volumeとmin_volumeの範囲内に収める
    return max(min(volume, max_volume), min_volume)


# 録画ファイルの保存ディレクトリ
output_dir = "videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 録画を開始する関数
def start_recording():
    # ファイル名を決定
    existing_files = os.listdir(output_dir)
    video_number = len([f for f in existing_files if f.startswith(f'{p_id}-{condition}-')]) + 1
    output_filename = os.path.join(output_dir, f'{p_id}-{condition}-{video_number}.mp4')
    
    # ビデオライターを初期化
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  #  エンコーダーを指定
    fps = 30.0  # フレームレート
    frame_size = (640, 480)  # カメラの解像度に合わせる
    video_writer = cv.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    return video_writer

# 変数を初期化
video_writer = None
is_recording = False

# run
camera_thread.start()
recognition_thread.start()

while True:
    frame = camera_thread.queue.get()

    # 録画中の場合はフレームを書き込む
    if is_recording and video_writer is not None:
        video_writer.write(frame)

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
    
    key = cv.waitKey(1)

    # 's' キーが押されたら録画を開始
    if key == ord('s'):
        if not is_recording:
            video_writer = start_recording()  # 録画を開始
            is_recording = True
            print("録画を開始しました")

    # 'e' キーが押されたら録画を終了
    elif key == ord('e'):
        if is_recording:
            video_writer.release()  # 録画を終了
            is_recording = False
            print("録画を終了しました")

    # 'q' キーが押されたら終了
    elif key == ord("q"):
        recognition_thread.stop()
        recognition_thread.in_queue.put(frame)
        camera_thread.stop()
        if is_recording:
            video_writer.release()  # 録画を終了
            is_recording = False #録画のフラグを終了
        break

cv.destroyAllWindows()