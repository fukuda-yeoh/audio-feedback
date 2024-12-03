import os
import cv2 as cv
import numpy as np

from audio_feedback.camera import RealSenseThread
from audio_feedback.recognition import HSVColorModel, RecognitionThread
import synthizer
from audio_feedback.defs import project_root
from audio_feedback.feedback_augment import calculate_volume

p_id = 1
condition = 1

a = np.array([[0, 0, 0]])

#Realsenseを開始
realsense_thread = RealSenseThread()

# モデルのHSVの設定
model = HSVColorModel(
    hue_range=(100, 130), saturation_range=(180, 240), value_range=(110, 255)
)

# 物体の認識やトラッキングを行うためのスレッドを生成するクラス
recognition_thread = RecognitionThread(model, realsense_thread.output_queue)

# 音のフィードバックの設定
synthizer.initialized()
context = synthizer.Context()
context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
context.default_distance_model.value = synthizer.DistanceModel.LINEAR

# 音
sound_file = project_root() / "sound_files" / "droplet.wav"
buffer = synthizer.Buffer.from_file(str(sound_file))
generator = synthizer.BufferGenerator(context)
generator.buffer.value = buffer
generator.looping.value = True

source = synthizer.Source3D(context)
source.add_generator(generator)
source.play()
source_sound_on = True

# 録画ファイルの保存ディレクトリ
output_dir = "videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# 動画の開始
def start_recording():
    existing_files = os.listdir(output_dir)
    video_number = (
        len([f for f in existing_files if f.startswith(f"{p_id}-{condition}-")]) + 1
    )
    output_filename = os.path.join(output_dir, f"{p_id}-{condition}-{video_number}.mp4")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")  # エンコーダーを指定
    fps = 30.0  # フレームレート
    frame_size = (640, 480)  # カメラの解像度に合わせる
    video_writer = cv.VideoWriter(output_filename, fourcc, fps, frame_size)

    return video_writer


# Initialize variables
video_writer = None
is_recording = False

# Run
recognition_thread.start()
realsense_thread.start()

try:
    while True:
        # 物体認識の結果処理
        color_frame, depth_frame, result = recognition_thread.out_queue.get()
        if color_frame is None:
            continue

        color_image = realsense_thread.convert_to_array(color_frame)
        # If recording, write the frame to the video file
        if is_recording and video_writer is not None:
            video_writer.write(color_image)

        if result:
            center = result.center
            img_width, img_height, _ = color_image.shape  # カラー画像のサイズ取得

            # # Calculate panning and pitch based on ball's position
            # pan = calculate_pan(center[0], img_width)
            # pitch = calculate_pitch(center[1], img_height)

            # # Apply panning (left-right) and pitch (up-down)
            # source.pan = pan
            # source.pitch = pitch

            median_depth = realsense_thread.get_median_depth(center, 5, depth_frame)
            x, y, z = realsense_thread.convert_to_3d(depth_frame, median_depth, center)

            # Set the ball's position based on its center
            ball_position = (-x, -y, -z)
            source.position.value = ball_position

            # 距離と音量計算
            # distance = np.linalg.norm(
            #     np.array(ball_position) - np.array([0,0,0])
            # )
            # volume = calculate_volume(distance)
            # source.gain.value = volume

            # list = [[x, y, z]]
            # a = np.append(a, list, axis=0)

            # CSVデータ更新
            a = np.append(a, [[x, y, z]], axis=0)

            center_x = int(center[0])
            center_y = int(center[1])

            # 画像に3D座標を注釈として表示
            cv.putText(
                color_image,
                f"Position: ({x:.2f}, {y:.2f}, {z:.2f})",
                (center_x, center_y - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

            # 物体認識の確認（緑の丸）
            cv.circle(color_image, (center_x, center_y), 10, (0, 255, 0), 2)

            if not source_sound_on:
                source.play()
                source_sound_on = True
        else:
            list = [[-1, -1, -1]]
            a = np.append(a, list, axis=0)
            source.stop()
            source_sound_on = False

        # Display the camera feed with annotations
        cv.imshow("RealSense Camera", color_image)

        key = cv.waitKey(1)

        # 's' to start recording
        if key == ord("d"):
            if not is_recording:
                video_writer = start_recording()
                is_recording = True
                print("録画を開始しました")

        # 'e' to end recording
        elif key == ord("e"):
            if is_recording:
                video_writer.release()
                is_recording = False
                print(a)
                np.savetxt(f"./aaa.csv", a, delimiter=",", fmt="%.3f")
                print("録画を終了しました")

        # 'q' to quit
        elif key == ord("q"):
            recognition_thread.stop()
            recognition_thread.in_queue.put(color_image)
            if is_recording:
                video_writer.release()
                is_recording = False
            break

    # np.savetxt(f'../aaa.csv', a, delimiter=',', fmt="%.3f")


finally:
    # Stop the pipeline and close windows
    cv.destroyAllWindows()
    synthizer.shutdown()
