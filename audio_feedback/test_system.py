# main_stereo_audio.py (YOLO + RealSense + Synthizer + 映像表示版)

import pyrealsense2 as rs
import numpy as np
import threading
from queue import Queue, Empty
import cv2 as cv
import time
import os
import synthizer  # 立体音響ライブラリ

# --- 外部モジュールのインポート ---
# スレッドクラスとユーティリティ関数（RealSenseThreadにパイプライン管理を任せる設計）
from realsense_thread import RealSenseThread
from yolo_thread import YOLOThread


# プロジェクトルートパスの代わり
def project_root():
    # 実際のプロジェクト構造に合わせて修正してください
    return os.path.dirname(os.path.abspath(__file__))


# ----------------------------------------------------------------------
# ユーティリティ関数（元のコードから移植）
# ----------------------------------------------------------------------


# 動画の開始 (フレームサイズをRealSenseの640x480に固定)
def start_recording():
    output_dir = "videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    p_id = 1
    condition = 1
    existing_files = os.listdir(output_dir)
    video_number = (
        len([f for f in existing_files if f.startswith(f"{p_id}-{condition}-")]) + 1
    )
    output_filename = os.path.join(output_dir, f"{p_id}-{condition}-{video_number}.mp4")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    frame_size = (640, 480)
    video_writer = cv.VideoWriter(output_filename, fourcc, fps, frame_size)
    return video_writer


# ----------------------------------------------------------------------
# メインシステム
# ----------------------------------------------------------------------


def main_system():

    yolo_thread = None
    rs_thread = None
    video_writer = None
    is_recording = False

    # --- 1. RealSenseスレッドのセットアップとキューの連携 ---

    rs_thread = RealSenseThread()
    print("[MAIN] RealSense Thread initialized.")

    # --- 2. Synthizer 初期化 ---
    # synthizer.initialize()
    # context = synthizer.Context()
    # context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
    # context.default_distance_model.value = synthizer.DistanceModel.LINEAR

    # sound_file = os.path.join(project_root(), "sound_files", "droplet.wav")
    # if not os.path.exists(sound_file):
    #     print(
    #         f"[MAIN] WARNING: Sound file not found at {sound_file}. Ensure path is correct."
    #     )

    # buffer = synthizer.Buffer.from_file(str(sound_file))
    # generator = synthizer.BufferGenerator(context)
    # generator.buffer.value = buffer
    # generator.looping.value = True

    # source = synthizer.Source3D(context)
    # source.add_generator(generator)
    # # source.play() は検出時に行う
    # source_sound_on = False

    # --- 3. スレッドの起動 ---
    yolo_thread = YOLOThread(
        input_queue=rs_thread.output_queue,
    )
    try:
        yolo_thread.start()
        print("[MAIN] YOLO Thread started.")

        while yolo_thread.ready == False:
            time.sleep(1)

        rs_thread.start()
        print("[MAIN] RealSense Thread started.")
    except Exception as e:
        print(e)

        # --- 4. メインループ (映像表示, 描画, 音響フィードバック) ---
        print(
            "[MAIN] System running. Press 'q' or 'e' (end record), 'd' (start record) to control."
        )

    while True:
        color_image, depth_frame, my_results = yolo_thread.output_queue.get()

        # if my_results:
        #     (x1, y1, x2, y2), conf, label = my_results[0]

        #     cx = int((x1 + x2) / 2)
        #     cy = int((y1 + y2) / 2)
        #     img_width, img_height, _ = color_image.shape  # カラー画像のサイズ取得

        #     median_depth = rs_thread.get_median_depth((cx, cy), 5, depth_frame)
        #     x, y, z = rs_thread.convert_to_3d(depth_frame, median_depth, (cx, cy))

        #     cv.putText(
        #         color_image,
        #         f"Position: ({x:.2f}, {y:.2f}, {z:.2f})",
        #         (cx - 100, cy - 20),
        #         cv.FONT_HERSHEY_SIMPLEX,
        #         0.5,
        #         (255, 0, 0),
        #         2,
        #     )
        #     cv.circle(color_image, (cx, cy), 10, (0, 255, 0), 2)

        #     # 立体音響ロジック
        #     ball_position = (x, y, z)
        #     source.position.value = ball_position

        #     if not source_sound_on:
        #         source.play()
        #         source_sound_on = True

        #     print(
        #         f"[MAIN] Detected: {label} ({conf:.2f}) | Pos(m): X={x:.2f}, Y={y:.2f}, Z={z:.2f} \r",
        #         end="",
        #     )

        # else:
        #     # 検出結果がない場合
        #     source.pause()
        #     source_sound_on = False
        #     print("[MAIN] Detecting... (No object detected) \r", end="")

        # # 録画
        # if is_recording and video_writer is not None:
        #     video_writer.write(color_image)

        # 画像の表示
        cv.imshow("RealSense Camera", color_image)

        # キー入力処理 (元のコードから移植)
        key = cv.waitKey(1)
        # ... (キー入力処理 'd', 'e', 'q' のロジックは省略) ...

        # if key == ord("d"):
        #     if not is_recording:
        #         video_writer = start_recording()
        #         is_recording = True
        #         print("録画を開始しました")
        # elif key == ord("e"):
        #     if is_recording:
        #         video_writer.release()
        #         is_recording = False
        #         # CSV保存ロジックはデータ構造 a が定義されていないため省略
        #         print("録画を終了しました")
        if key == ord("q"):
            # RealSenseとYOLOスレッドを停止する
            break

    # --- 6. 終了処理 ---
    print("\n[MAIN] Stopping all threads and streams...")

    if rs_thread:
        rs_thread.stop()

    if yolo_thread:
        yolo_thread.stop()

        yolo_thread.join()
        rs_thread.join()

    if is_recording and video_writer:
        video_writer.release()

    # プログラム終了時に音源を確実に停止させる
    if source_sound_on:
        source.pause()  # stop() の代わりに pause() を使用

    cv.destroyAllWindows()
    synthizer.shutdown()
    print("[MAIN] System cleanup complete.")


if __name__ == "__main__":
    main_system()
