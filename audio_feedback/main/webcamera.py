import os

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv

from audio_feedback.camera import CameraThread, load_intrinsic, load_extrinsic
from audio_feedback.defs import project_root
from audio_feedback.recognition import HSVColorModel, RecognitionThread


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

# 録画ファイルの保存ディレクトリ
output_dir = "videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 録画を開始する関数
def start_recording():
    # ファイル名を決定
    existing_files = os.listdir(output_dir)
    video_number = (
        len([f for f in existing_files if f.startswith(f"{p_id}-{condition}-")]) + 1
    )
    output_filename = os.path.join(output_dir, f"{p_id}-{condition}-{video_number}.mp4")

    # ビデオライターを初期化
    fourcc = cv.VideoWriter_fourcc(*"mp4v")  #  エンコーダーを指定
    fps = 30.0  # フレームレート
    frame_size = (640, 480)  # カメラの解像度に合わせる
    video_writer = cv.VideoWriter(output_filename, fourcc, fps, frame_size)

    return video_writer


# 変数を初期化
video_writer = None
is_recording = False

# run
camera_thread.start()

while True:
    frame = camera_thread.queue.get()

    # 録画中の場合はフレームを書き込む
    if is_recording and video_writer is not None:
        video_writer.write(frame)

    cv.imshow(f"Camera {camera_no}", frame)

    key = cv.waitKey(1)

    # 's' キーが押されたら録画を開始
    if key == ord("s"):
        if not is_recording:
            video_writer = start_recording()  # 録画を開始
            is_recording = True
            print("録画を開始しました")

    # 'e' キーが押されたら録画を終了
    elif key == ord("e"):
        if is_recording:
            video_writer.release()  # 録画を終了
            is_recording = False
            print("録画を終了しました")

    # 'q' キーが押されたら終了
    elif key == ord("q"):
        camera_thread.stop()
        if is_recording:
            video_writer.release()  # 録画を終了
            is_recording = False  # 録画のフラグを終了
        break

cv.destroyAllWindows()
