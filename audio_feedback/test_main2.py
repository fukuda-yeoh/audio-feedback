import os
import pyrealsense2 as rs
import cv2 as cv
import numpy as np
import time

from audio_feedback.recognition import HSVColorModel, RecognitionThread
from audio_feedback.tones import Listener, Sound, Source
from audio_feedback.defs import project_root

p_id = 1
condition = 1

# RealSense pipeline setup
pipeline = rs.pipeline()
config = rs.config()

# Enable depth and color streams
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# モデルのHSVの設定
model = HSVColorModel(
    hue_range=(100, 130), saturation_range=(180, 240), value_range=(110, 255)
)

# 物体の認識やトラッキングを行うためのスレッドを生成するクラス
recognition_thread = RecognitionThread(model)

# 音のフィードバックの設定
listener = Listener()
source = Source()

# Initialise sound
sound_file = project_root() / "sound_files" / "droplet.wav"
my_sound = Sound(sound_file)

# Set listener and source positions (初期位置)
listener.position = (0, 240, 0)  # 画面左端の中央
listener.orientation = ((-1.0, 0.0, 0.0), (0.0, 0.0, 1.0))  # x方向(右方向)，z方向（上方向）を向いている

# Load sound into source
source.add_sound(my_sound)
source.loop = True
source.rolloff = 0.05  # 音量の減衰の仕方を決める
source.play()
source_sound_on = True

# Function to calculate volume based on distance
def calculate_volume(distance, reference_distance=1.0, max_volume=200.0, min_volume=20.0):
    if distance < reference_distance:
        distance = reference_distance
    volume = max_volume / (distance / reference_distance) ** 2
    return max(min(volume, max_volume), min_volume)

# 録画ファイルの保存ディレクトリ
output_dir = "videos"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 動画の開始
def start_recording():
    existing_files = os.listdir(output_dir)
    video_number = len([f for f in existing_files if f.startswith(f'{p_id}-{condition}-')]) + 1
    output_filename = os.path.join(output_dir, f'{p_id}-{condition}-{video_number}.mp4')
    
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # エンコーダーを指定
    fps = 30.0  # フレームレート
    frame_size = (640, 480)  # カメラの解像度に合わせる
    video_writer = cv.VideoWriter(output_filename, fourcc, fps, frame_size)
    
    return video_writer

# Initialize variables
video_writer = None
is_recording = False

# Run
recognition_thread.start()

try:
    while True:
        # Wait for frames from the RealSense camera
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame() # カラー画像
        depth_frame = frames.get_depth_frame() # 深度画像

        if not color_frame or not depth_frame:
            continue

        # データをNumpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # If recording, write the frame to the video file
        if is_recording and video_writer is not None:
            video_writer.write(color_image)

        # 物体認識の結果処理
        recognition_thread.in_queue.put(color_image)
        result = recognition_thread.out_queue.get()

        if result:
            center = result.center
            img_width, img_height, _ = color_image.shape #カラー画像のサイズ取得
            
            # Set the ball's position based on its center
            ball_position = (center[0], center[1], 0)
            source.position = ball_position

            # Compute the distance between the listener and the ball
            distance = np.linalg.norm(np.array(ball_position[:2]) - np.array(listener.position[:2]))
            volume = calculate_volume(distance)
    
            # Set the volume
            source.volume = volume

            # Get the depth at the center of the object (convert center coordinates to integers)
            center_x = int(center[0])
            center_y = int(center[1])

            window_size = 5
            window = (
                depth_image[
                    int(center_y - window_size / 2) : int(center_y + window_size / 2),
                    int(center_x - window_size / 2) : int(center_x + window_size / 2),
                ]
                * depth_frame.get_units()
            )
            window[window == 0] = np.nan

            # Calculate the median depth from valid points
            median_depth = np.nanmedian(window)

            # Print out the depth value
            print(f"Depth value (mm): {median_depth}")

            # Annotate depth on the image
            cv.putText(
                color_image, 
                f"Depth: {median_depth:.2f} m", 
                (center_x, center_y - 10), 
                cv.FONT_HERSHEY_SIMPLEX, 
                0.9, 
                (255, 0, 0), 
                2
            )

            # Draw a circle at the center of the recognized object
            #cv.circle(color_image, (center_x, center_y), 10, (0, 255, 0), 2)

            # Draw a rectangle around the recognized object (assuming a 50x50 size for demonstration)
            cv.rectangle(color_image, (center_x - 25, center_y - 25), (center_x + 25, center_y + 25), (0, 255, 0), 2)

            if not source_sound_on:
                source.play()
                source_sound_on = True
        else:
            source.stop()
            source_sound_on = False


        # Display the camera feed with annotations
        cv.imshow("RealSense Camera", color_image)
        
        key = cv.waitKey(1)

        # 's' to start recording
        if key == ord('s'):
            if not is_recording:
                video_writer = start_recording()
                is_recording = True
                print("録画を開始しました")

        # 'e' to end recording
        elif key == ord('e'):
            if is_recording:
                video_writer.release()
                is_recording = False
                print("録画を終了しました")

        # 'q' to quit
        elif key == ord("q"):
            recognition_thread.stop()
            recognition_thread.in_queue.put(color_image)
            if is_recording:
                video_writer.release()
                is_recording = False
            break

finally:
    # Stop the pipeline and close windows
    pipeline.stop()
    cv.destroyAllWindows()
