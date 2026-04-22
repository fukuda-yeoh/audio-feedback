# multicamera.py (Step 1 + CSV Logging)

import pyrealsense2 as rs
import numpy as np
import threading
from queue import Queue, Empty
import cv2 as cv
import time
import os
import synthizer
import csv
import datetime

# --- 外部モジュールのインポート ---
from realsense_thread import RealSenseThread
from yolo_thread import YOLOThread
from base_conversion import CameraTransformer

# ==============================================================================
# 【設定エリア】 カメラの個体識別と配置設定
# ==============================================================================

# シリアルナンバー定義
SERIAL_LEFT  = "845112070212"
SERIAL_RIGHT = "147122071512"

# 各カメラの配置設定
CAMERA_SETTINGS = {
    SERIAL_LEFT: {
        "id": "Left",
        "offset": [-0.1, 0.0, 0.0],     # 左に10cm
        "rotation": [0.0, -30.0, 0.0]   # 左(外)向きに30度
    },
    SERIAL_RIGHT: {
        "id": "Right",
        "offset": [0.1, 0.0, 0.0],      # 右に10cm
        "rotation": [0.0, 30.0, 0.0]    # 右(外)向きに30度
    }
}

# ==============================================================================

def project_root():
    return os.path.dirname(os.path.abspath(__file__))

def find_realsense_serials():
    ctx = rs.context()
    devices = ctx.query_devices()
    serials = []
    for dev in devices:
        serials.append(dev.get_info(rs.camera_info.serial_number))
    return serials

def start_recording(camera_id):
    output_dir = "videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_filename = os.path.join(output_dir, f"{camera_id}-{time.strftime('%Y%m%d_%H%M%S')}.mp4")
    fourcc = cv.VideoWriter_fourcc(*"mp4v")
    return cv.VideoWriter(output_filename, fourcc, 30.0, (640, 480))

# --- CSVログ機能の追加 ---
def create_csv_logger():
    """CSVファイルを作成し、ライターオブジェクトを返す"""
    log_dir = "logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # ファイル名に日時を入れる
    filename = os.path.join(log_dir, f"tracking_log_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    
    # ファイルオープン (newline='' はCSVモジュールで必須)
    csv_file = open(filename, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    
    # ヘッダー書き込み
    header = [
        "Timestamp", "Camera_ID", "Label", "Confidence",
        "User_X", "User_Y", "User_Z",  # 統合座標
        "Raw_X", "Raw_Y", "Raw_Z"      # 生座標(参考用)
    ]
    writer.writerow(header)
    
    print(f"[MAIN] CSVログ記録開始: {filename}")
    return csv_file, writer

# ----------------------------------------------------------------------
# メインシステム
# ----------------------------------------------------------------------

def main_system():
    camera_systems = {}
    video_writers = {}
    is_recording = False
    source_sound_on = False

    # --- 1. デバイス検出 ---
    connected_serials = find_realsense_serials()
    print(f"[MAIN] 接続されたデバイス: {connected_serials}")

    if not connected_serials:
        print("[MAIN] エラー: カメラが見つかりません。")
        return

    # --- 2. Synthizer & CSV 初期化 ---
    # CSVロガーの準備
    csv_file, csv_writer = create_csv_logger()

    try: # tryブロック開始 (エラー終了時にCSVを閉じるため)
        synthizer.initialize()
        context = synthizer.Context()
        context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
        context.default_distance_model.value = synthizer.DistanceModel.LINEAR

        sound_path = os.path.join(project_root(), "sound_files", "droplet.wav")
        buffer = synthizer.Buffer.from_file(sound_path)
        generator = synthizer.BufferGenerator(context)
        generator.buffer.value = buffer
        generator.looping.value = True
        source = synthizer.Source3D(context)
        source.add_generator(generator)
        
        source.distance_model.value = synthizer.DistanceModel.EXPONENTIAL
        source.rolloff.value = 1.0
        source.distance_ref.value = 0.4
        source.distance_max.value = 3.2


        # --- 3. システム構築 ---
        for serial in connected_serials:
            if serial in CAMERA_SETTINGS:
                settings = CAMERA_SETTINGS[serial]
                cam_id = settings["id"]
                print(f"\n[MAIN] セットアップ中: {cam_id} Camera (Serial: {serial})")
                
                transformer = CameraTransformer(settings["offset"], settings["rotation"])
                
                rs_thread = RealSenseThread(serial_number=serial, name=f"RS-{cam_id}")
                yolo_thread = YOLOThread(input_queue=rs_thread.output_queue, name=f"YOLO-{cam_id}")

                camera_systems[serial] = {
                    "id": cam_id,
                    "rs_thread": rs_thread,
                    "yolo_thread": yolo_thread,
                    "output_queue": yolo_thread.output_queue,
                    "transformer": transformer
                }

                yolo_thread.start()
                while not yolo_thread.ready: time.sleep(0.1)
                rs_thread.start()
            else:
                print(f"[MAIN] 警告: 未登録のカメラ (Serial: {serial}) はスキップします。")


        # --- 4. メインループ ---
        print("\n[MAIN] システム稼働開始。's':録画, 'e':停止, 'q':終了")

        while True:
            processed_any = False
            
            for serial, system in camera_systems.items():
                queue = system["output_queue"]
                cam_id = system["id"]
                transformer = system["transformer"]
                rs_thread = system["rs_thread"]

                try:
                    color_image, depth_frame, results = queue.get_nowait()
                    processed_any = True

                    if results:
                        (x1, y1, x2, y2), conf, label = results[0]
                        cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

                        # --- A. 3D座標取得 ---
                        w, h = x2 - x1, y2 - y1
                        median_depth = rs_thread.get_median_depth((cx, cy), w, h, depth_frame)
                        cam_x, cam_y, cam_z = rs_thread.convert_to_3d(depth_frame, median_depth, (cx, cy))

                        # --- B. 座標変換 ---
                        user_x, user_y, user_z = transformer.transform_to_head_coords([cam_x, cam_y, cam_z])

                        # --- CSV書き込み (ここに追加) ---
                        # フォーマット: [Time, ID, Label, Conf, Ux, Uy, Uz, Rx, Ry, Rz]
                        current_time = datetime.datetime.now().strftime("%H:%M:%S.%f")
                        csv_writer.writerow([
                            current_time, cam_id, label, f"{conf:.2f}",
                            f"{user_x:.3f}", f"{user_y:.3f}", f"{user_z:.3f}",
                            f"{cam_x:.3f}", f"{cam_y:.3f}", f"{cam_z:.3f}"
                        ])

                        # --- C. 描画とフィードバック ---
                        text = f"USER: X{user_x:.2f}, Y{user_y:.2f}, Z{user_z:.2f} | {label}"
                        cv.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        cv.putText(color_image, text, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

                        source.position.value = (user_x, -user_y, -user_z)

                        if not source_sound_on:
                            source.play()
                            source_sound_on = True
                        
                        print(f"[{cam_id}] {label}: USER({user_x:.2f}, {user_y:.2f}, {user_z:.2f}) \r", end="")

                    if is_recording and serial in video_writers:
                        video_writers[serial].write(color_image)
                    
                    cv.imshow(f"Camera: {cam_id}", color_image)

                except Empty:
                    pass
                except Exception as e:
                    print(f"Error in {cam_id}: {e}")

            if not processed_any and source_sound_on:
                pass 

            key = cv.waitKey(1)
            if key == ord("q"): break
            elif key == ord("s") and not is_recording:
                for s in camera_systems: video_writers[s] = start_recording(camera_systems[s]["id"])
                is_recording = True
                print("\nRecording Started")
            elif key == ord("e") and is_recording:
                for w in video_writers.values(): w.release()
                video_writers = {}
                is_recording = False
                print("\nRecording Stopped")

    finally:
        print("\n[MAIN] 終了処理中...")
        # CSVファイルを閉じる
        if csv_file:
            csv_file.close()
            print("[MAIN] CSVログ保存完了")

        for sys in camera_systems.values():
            sys["rs_thread"].stop()
            sys["yolo_thread"].stop()
        for sys in camera_systems.values():
            sys["rs_thread"].join()
            sys["yolo_thread"].join()
        if is_recording:
            for w in video_writers.values(): w.release()
        synthizer.shutdown()
        cv.destroyAllWindows()

if __name__ == "__main__":
    main_system()