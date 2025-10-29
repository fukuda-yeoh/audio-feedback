# multicamera.py (修正後のメインコード全体)

import pyrealsense2 as rs
import numpy as np
import threading
from queue import Queue, Empty
import cv2 as cv
import time
import os
import synthizer  # 立体音響ライブラリ

# --- 外部モジュールのインポート ---
from realsense_thread import RealSenseThread
from yolo_thread import YOLOThread

# カメラの識別子（仮）
CAMERA_IDS = ['cam_left', 'cam_right']

# プロジェクトルートパスの代わり
def project_root():
    # 実際のプロジェクト構造に合わせて修正してください
    return os.path.dirname(os.path.abspath(__file__))

# ----------------------------------------------------------------------
# ユーティリティ関数
# ----------------------------------------------------------------------

# --- カメラのシリアル番号を検出するユーティリティ ---
def find_realsense_serials(num_devices=2):
    """接続されているRealSenseデバイスのシリアル番号を検出する"""
    context = rs.context()
    devices = context.query_devices()
    serials = []
    
    # 接続されているデバイスを列挙し、シリアル番号を取得
    for i, device in enumerate(devices):
        if i >= num_devices:
            break
        serial = device.get_info(rs.camera_info.serial_number)
        serials.append(serial)
    
    # 必要な台数に満たない場合はエラー
    if len(serials) < num_devices:
        raise RuntimeError(f"Required {num_devices} RealSense cameras, but found only {len(serials)}. Please connect two cameras.")
        
    print(f"[MAIN] Detected RealSense Serials: {serials}")
    return serials

# 動画の開始 (フレームサイズをRealSenseの640x480に固定)
def start_recording(camera_id):
    output_dir = "videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    p_id = 1
    condition = 1
    # カメラIDをファイル名に含める
    output_filename = os.path.join(output_dir, f"{p_id}-{condition}-{camera_id}-{time.strftime('%Y%m%d_%H%M%S')}.mp4")

    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    fps = 30.0
    frame_size = (640, 480)
    video_writer = cv.VideoWriter(output_filename, fourcc, fps, frame_size)
    return video_writer


# ----------------------------------------------------------------------
# メインシステム
# ----------------------------------------------------------------------


def main_system():

    # 各カメラのスレッドとリソースを格納する辞書
    camera_systems = {}
    video_writers = {}
    is_recording = False
    
    source_sound_on = False

    # --- 1. RealSenseデバイスの検出 ---
    try:
        device_serials = find_realsense_serials(len(CAMERA_IDS))
    except RuntimeError as e:
        print(f"[MAIN] ERROR: {e}")
        return
        
    # カメラIDとシリアル番号をマッピング
    camera_map = dict(zip(CAMERA_IDS, device_serials))

    # --- 2. Synthizer 初期化 ---
    synthizer.initialize()
    context = synthizer.Context()
    context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
    context.default_distance_model.value = synthizer.DistanceModel.LINEAR

    sound_file = os.path.join(project_root(), "sound_files", "droplet.wav")
    if not os.path.exists(sound_file):
        print(
            f"[MAIN] WARNING: Sound file not found at {sound_file}. Ensure path is correct."
        )

    buffer = synthizer.Buffer.from_file(str(sound_file))
    generator = synthizer.BufferGenerator(context)
    generator.gain.value = 1
    generator.pitch_bend.value = 1
    generator.buffer.value = buffer
    generator.looping.value = True

    source = synthizer.Source3D(context)
    source.add_generator(generator)
    # source.play() は検出時に行う
    source.distance_model.value = synthizer.DistanceModel.EXPONENTIAL # モデルを指定する
    source.rolloff.value = 1.0  # 減衰の強さ
    source.distance_ref.value = 0.4  # 音量が最大となる基準距離
    source.distance_max.value = 3.2  # 音が聞こえる最大距離
    source_sound_on = True


    # --- 3. RealSense & YOLO スレッドペアのセットアップと起動 ---
    for cam_id in CAMERA_IDS:
        print(f"\n[MAIN] Setting up system for {cam_id}...")
        
        serial_num = camera_map[cam_id]
        
        # 3-A. RealSense スレッドの初期化 (serial_numberを渡す)
        rs_thread = RealSenseThread(
            serial_number=serial_num, 
            name=f"RS-{cam_id}"
        )
        
        # 3-B. YOLO スレッドの初期化 (以前のTypeError修正のため nameを渡す)
        yolo_thread = YOLOThread(
            input_queue=rs_thread.output_queue, 
            name=f"YOLO-{cam_id}",
        )

        # 3-C. スレッドの格納
        camera_systems[cam_id] = {
            'rs_thread': rs_thread,
            'yolo_thread': yolo_thread,
            'yolo_output_queue': yolo_thread.output_queue # メインループでの取得用
        }

        # 3-D. スレッドの起動
        yolo_thread.start()
        print(f"[{cam_id}] YOLO Thread started. Waiting for ready...")
        
        # YOLOThreadが準備できるまで待機
        while not yolo_thread.ready:
            time.sleep(0.1)
        
        rs_thread.start()
        print(f"[{cam_id}] RealSense Thread started.")


    # --- 4. メインループ (映像表示, 描画, 音響フィードバック) ---
    print(
        "\n[MAIN] System running. Press 'q' or 'e' (end record), 's' (start record) to control."
    )
    
    # cam_index の制御は削除

    while True:
        processed_any_frame = False
        
        # すべてのカメラシステムをループでチェックする
        for cam_id in CAMERA_IDS:
            system = camera_systems[cam_id]
            yolo_output_queue = system['yolo_output_queue']
            rs_thread_instance = system['rs_thread']

            try:
                # YOLOスレッドから処理結果を**非ブロッキング**で取得
                # キューにデータがあれば、get_nowait()で即座に取り出す
                color_image, depth_frame, my_results = yolo_output_queue.get_nowait()
                
                processed_any_frame = True # 少なくとも一つのフレームを処理した
                current_detection = None
                
                if my_results:
                    # 検出結果の処理 (最も信頼度の高いものを使用)
                    (x1, y1, x2, y2), conf, label = my_results[0]
                    current_detection = my_results[0]
                    
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    
                    # --- 3D座標計算 (RealSenseThreadのメソッドを使用) ---
                    bbox_w = x2 - x1
                    bbox_h = y2 - y1
                    median_depth = rs_thread_instance.get_median_depth((cx, cy), bbox_w, bbox_h, depth_frame)
                    
                    # Intrinsicsを取得し、3D座標に変換
                    # RealSenseThreadのconvert_to_3dメソッドを使用
                    x, y, z = rs_thread_instance.convert_to_3d(depth_frame, median_depth, (cx, cy))
                    
                    # --- 描画 (3D座標) ---
                    cv.putText(
                        color_image,
                        f"Pos({x:.2f}, {y:.2f}, {z:.2f}) | {label}",
                        (int(x1), int(y1) - 10),
                        cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1
                    )
                    cv.circle(color_image, (cx, cy), 5, (0, 255, 0), -1)

                    # Set the ball's position based on its center
                    ball_position = (x, -y, -z)
                    source.position.value = ball_position

                    if not source_sound_on:
                        source.play()
                        source_sound_on = True

                    print(
                        f"[MAIN][{cam_id}] Detected: {label} ({conf:.2f}) | Pos(m): X={x:.2f}, Y={y:.2f}, Z={z:.2f} \r",
                        end="",
                    )
                else:
                    # 検出結果がない場合
                    print(f"[MAIN][{cam_id}] Detecting... (No object detected) \r", end="")
                
                # --- 画像の表示と録画 ---
                if is_recording and cam_id in video_writers:
                    video_writers[cam_id].write(color_image)
                
                cv.imshow(f"RealSense Camera - {cam_id}", color_image)
                
            except Empty:
                # キューが空の場合はスキップし、次のカメラのキューをチェック
                pass
            except Exception as e:
                print(f"[MAIN] Error in main loop for {cam_id}: {e}")

        # -------------------------------------------------------------
        # 音響フィードバックの停止制御 (全てのキューが空で、かつ音源が再生中の場合のみ停止)
        # -------------------------------------------------------------
        if not processed_any_frame and source_sound_on:
            # Note: ここは簡易化のために、前のフレームの処理結果に基づいて音を鳴らし続けています。
            # 厳密には、全てのキューが空になったときにのみ停止すべきです。
            all_queues_empty = True
            for cam_id in CAMERA_IDS:
                if not camera_systems[cam_id]['yolo_output_queue'].empty():
                    all_queues_empty = False
                    break
            
            if all_queues_empty:
                source.pause()
                source_sound_on = False


        # キー入力処理 (全カメラのチェック後に一度だけ実行)
        key = cv.waitKey(1)

        if key == ord("s"):
            if not is_recording:
                print("\n[DEBUG] Attempting to start recording for all cameras...")
                for cid in CAMERA_IDS:
                    video_writers[cid] = start_recording(cid)
                    print(f"[DEBUG] Started video writer for {cid}")
                is_recording = True
                print("\n録画を開始しました")
                
        elif key== ord("e"):
            if is_recording:
                for writer in video_writers.values():
                    writer.release()
                video_writers = {}
                is_recording = False
                print("\n録画を終了しました")
        elif key == ord("q"):
            break
        
    # --- 5. 終了処理 ---
    print("\n[MAIN] Stopping all threads and streams...")

    for system in camera_systems.values():
        system['rs_thread'].stop()
        system['yolo_thread'].stop()
    
    # スレッドが終了するのを待つ
    for system in camera_systems.values():
        system['rs_thread'].join()
        system['yolo_thread'].join()

    if is_recording:
        for writer in video_writers.values():
            writer.release()

    if source_sound_on:
        source.pause()

    cv.destroyAllWindows()
    synthizer.shutdown()
    print("[MAIN] System cleanup complete.")


if __name__ == "__main__":
    main_system()