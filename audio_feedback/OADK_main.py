import os
import cv2 as cv
import cv2
import numpy as np
import depthai as dai
import synthizer

 
from audio_feedback.camera.OAKD import OAKDThread
from audio_feedback.recognition import HSVColorModel, RecognitionThread
from audio_feedback.defs import project_root
from audio_feedback.feedback_augment import (
    calculate_point_source_gain,
    calculate_realistic_gain,
    calculate_gain,
    LINEAR_pitch,
    EXPONENTIAL_pitch,
    INVERSE_pitch,
)
 
p_id = 1
condition = 1
a = np.array([[0, 0, 0]])
 
# OAK-Dを開始する
oakd_thread = OAKDThread()
 
# モデルのHSVの設定
model = HSVColorModel(
    hue_range=(100, 230), saturation_range=(130,240),value_range=(220, 255)
)

# 物体の認識やトラッキングを行うためのスレッドを生成するクラス
recognition_thread = RecognitionThread(model, oakd_thread.output_queue)
 
# 音のフィードバックの設定
synthizer.initialize()
context = synthizer.Context()
context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
context.default_distance_model.value = synthizer.DistanceModel.EXPONENTIAL
 
sound_file = project_root() / "sound_files" / "1000Hz_v2.wav"
buffer = synthizer.Buffer.from_file(str(sound_file))
generator = synthizer.BufferGenerator(context)
generator.gain.value = 1
generator.pitch_bend.value = 1
generator.buffer.value = buffer
generator.looping.value = True
 
source = synthizer.Source3D(context)
source.add_generator(generator)
source.distance_model.value = synthizer.DistanceModel.EXPONENTIAL
source.rolloff.value = 1.0
source.distance_ref.value = 0.4
source.distance_max.value = 3.2
source.play()
source_sound_on = True


# 録画ファイルの保存ディレクトリ
output_dir = "videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
 
def start_recording():
    existing_files = os.listdir(output_dir)
    video_number = (
        len([f for f in existing_files if f.startswith(f"{p_id}-{condition}-")]) + 1
    )
    output_filename = os.path.join(output_dir, f"{p_id}-{condition}-{video_number}.mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    frame_size = (1280, 800)
    return cv.VideoWriter(output_filename, fourcc, fps, frame_size)
 
 
video_writer = None
is_recording = False
 
recognition_thread.start()
oakd_thread.start()

try:
    while True:
        # 物体認識の結果処理
        color_frame, depth_frame, result = recognition_thread.out_queue.get()
        if color_frame is None:
            continue
        

        color_image = color_frame.getCvFrame()
        # If recording, write the frame to the video file
        if is_recording and video_writer is not None:
            video_writer.write(color_image)

        if result:
            center = result.center
            img_width, img_height, _ = color_image.shape  # カラー画像のサイズ取得

            x, y, z = oakd_thread.get_spatial_coords(depth_frame, center)
            # 深度フレームを NumPy 配列に変換
            depth_array = depth_frame.getCvFrame()

            #以下のコードは修正用
             # 深度データの shape 確認
            depth_height, depth_width = depth_array.shape
            print(f"Depth frame size: {depth_width}x{depth_height}")

            # 深度データの確認
            print(depth_array.dtype, depth_array.min(), depth_array.max())

            center_x, center_y = int(center[0]), int(center[1])

            # インデックスが範囲内か確認
            if 0 <= center_x < depth_width and 0 <= center_y < depth_height:
                print(z, depth_array[center_y, center_x])  # z座標と元の深度値
            else:
                print(f"Warning: Center ({center_x}, {center_y}) is out of depth frame bounds.")
            #ここまで

            # Set the ball's position based on its center
            ball_position = (x, -y, -z)
            source.position.value = ball_position

            # 距離と音量計算
            distance = np.linalg.norm(np.array(ball_position) - np.array([0, 0, 0]))
            # pitch = LINEAR_pitch(
            #     distance,
            #     source.distance_ref.value,
            #     source.distance_max.value,
            # )
            
            pitch = EXPONENTIAL_pitch(
                distance,
                source.distance_ref.value,
            )
            # pitch = INVERSE_pitch(
            #     distance,
            #     source.distance_ref.value,
            # )

            generator.pitch_bend.value = pitch  
            # print(distance,generator.pitch_bend.value)

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
            source.pause()
            source_sound_on = False

         
        # 結果の映像を表示
        # cv2.imshow("Masked Frame", masked_frame)
        cv2.imshow("Original Frame", color_image)

        key = cv.waitKey(1)
        if key == ord("s"):
            if not is_recording:
                video_writer = start_recording()
                is_recording = True
                print("録画を開始しました")
        elif key == ord("e"):
            if is_recording:
                video_writer.release()
                is_recording = False
                print("録画を終了しました")
        elif key == ord("q"):
            recognition_thread.stop()
            oakd_thread.stop()
            if is_recording:
                video_writer.release()
                is_recording = False
            break
finally:
    cv.destroyAllWindows()
    synthizer.shutdown()