import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
import os

# --- カメラ設定 ---
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480
FRAME_RATE = 30 # 同期にはフレームレートが重要

# --- 記録設定 ---
# 記録画像を保存するディレクトリを絶対パスで指定
RECORDING_OUTPUT_DIR = r"C:\Users\oobuh\audio-feedback-yolo\recorded_frames" 

# --- 関数定義 ---

def find_realsense_devices():
    """
    接続されているRealSenseデバイスを検索し、シリアル番号のリストとコンテキストを返す。
    """
    ctx = rs.context()
    serials = []
    if len(ctx.devices) > 0:
        for dev in ctx.devices:
            s = dev.get_info(rs.camera_info.serial_number)
            serials.append(s)
            print(f"Found device: {dev.get_info(rs.camera_info.name)} (Serial: {s})")
    else:
        print("No Intel RealSense devices connected.")
    return serials, ctx

def setup_master_slave(ctx, serials):
    """
    検出されたデバイスをマスター/スレーブモードに設定する。
    少なくとも2台のカメラが必要。
    """
    if len(serials) < 2:
        print("ERROR: At least two devices are required for master-slave setup.")
        return False

    # デバイスオブジェクトをシリアル番号で取得
    devices = {dev.get_info(rs.camera_info.serial_number): dev for dev in ctx.devices}

    # 最初のデバイスをマスター、2番目のデバイスをスレーブとして設定
    master_serial = serials[0]
    slave_serial = serials[1]

    try:
        # マスターデバイスの設定
        master_dev = devices[master_serial]
        print(f"Setting device {master_serial} as master.")
        master_dev.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 1) # 1: Master

        # スレーブデバイスの設定
        slave_dev = devices[slave_serial]
        print(f"Setting device {slave_serial} as slave.")
        slave_dev.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 2) # 2: Slave
        return True
    except Exception as e:
        print(f"Failed to set master-slave sync mode: {e}")
        print("Ensure devices have depth sensors and are properly connected.")
        return False

def enable_device_streams(ctx, serials):
    """
    各カメラのパイプラインを設定し、カラーおよび深度ストリームを有効にする。
    """
    pipelines = {}
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        cfg.enable_stream(rs.stream.color, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.bgr8, FRAME_RATE)
        cfg.enable_stream(rs.stream.depth, RESOLUTION_WIDTH, RESOLUTION_HEIGHT, rs.format.z16, FRAME_RATE)
        pipe.start(cfg)
        pipelines[serial] = pipe
        print(f"Started pipeline for device: {serial}")
    
    # カメラが安定し、フレームを生成し始めるまで少し待機
    time.sleep(1.5) # 以前の1秒より少し長めに設定
    return pipelines

def check_synchronization(pipelines):
    """
    各カメラからフレームを取得し、タイムスタンプを比較して同期を確認する。
    キーボード入力に応じてフレームの記録を開始/停止する。
    """
    print("\n--- Checking Camera Synchronization ---")
    print("Press 's' to START recording frames.")
    print("Press 'e' to STOP recording frames.")
    print("Press 'q' or 'Esc' to quit.")

    align = rs.align(rs.stream.color) # 深度とカラーのアラインメント用
    
    is_recording = False # 記録状態を示すフラグ
    frame_counter = 0    # 記録中のフレームカウンタ

    # 記録ディレクトリが存在しない場合は作成
    os.makedirs(RECORDING_OUTPUT_DIR, exist_ok=True)

    try:
        while True:
            frame_timestamps = {}
            camera_images = {}
            current_frame_data = {} # 記録用: カラーと深度のNumPy配列を一時的に保持

            # 各パイプラインからフレームを取得
            for serial, pipe in pipelines.items():
                try:
                    frames = pipe.wait_for_frames(2000) # 2秒のタイムアウト
                    aligned_frames = align.process(frames)
                    color_frame = aligned_frames.get_color_frame()
                    depth_frame = aligned_frames.get_depth_frame()

                    if color_frame and depth_frame:
                        # カラーフレームのタイムスタンプを取得 (ミリ秒単位)
                        timestamp = color_frame.get_timestamp()
                        frame_timestamps[serial] = timestamp
                        
                        # 表示用に画像を変換
                        color_image = np.asanyarray(color_frame.get_data())
                        depth_image = np.asanyarray(depth_frame.get_data()) 
                        
                        camera_images[serial] = color_image
                        current_frame_data[serial] = {
                            "color": color_image,
                            "depth": depth_image # 深度データは取得するが、保存はしない
                        }
                    else:
                        print(f"Warning: Could not get aligned color/depth frame from device {serial}.")
                        frame_timestamps[serial] = -1 # 取得失敗を示す値
                        # フレームが取得できなかった場合でも、表示用に黒い画像を生成
                        camera_images[serial] = np.zeros((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3), dtype=np.uint8) 
                        current_frame_data[serial] = None # 記録しない

                except RuntimeError as e:
                    print(f"RuntimeError for device {serial}: {e}. Skipping frame for this camera.")
                    frame_timestamps[serial] = -1
                    camera_images[serial] = np.zeros((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3), dtype=np.uint8)
                    current_frame_data[serial] = None
                except Exception as e:
                    print(f"An unexpected error occurred for device {serial}: {e}. Skipping frame for this camera.")
                    frame_timestamps[serial] = -1
                    camera_images[serial] = np.zeros((RESOLUTION_HEIGHT, RESOLUTION_WIDTH, 3), dtype=np.uint8)
                    current_frame_data[serial] = None

            # タイムスタンプの比較と表示
            if len(frame_timestamps) == 2 and -1 not in frame_timestamps.values():
                serials_list = list(frame_timestamps.keys())
                ts1 = frame_timestamps[serials_list[0]]
                ts2 = frame_timestamps[serials_list[1]]
                time_diff = abs(ts1 - ts2)
                
                print(f"Frame Timestamps: {serials_list[0]}: {ts1:.2f}ms, {serials_list[1]}: {ts2:.2f}ms | Diff: {time_diff:.2f}ms", end=' ')
                
                if time_diff < 10: # 例: 10ms以内を同期とみなす
                    print("--> Cameras appear to be synchronized! (Difference < 10ms)")
                else:
                    print("--> Synchronization may be off. (Difference >= 10ms)")
            elif len(frame_timestamps) > 0:
                print("Waiting for frames from all cameras...")
            else:
                print("No cameras are streaming frames.")


            # 画像の表示
            displayed_image = None
            if len(camera_images) > 0:
                # 少なくとも1つのカメラ画像が利用可能であれば、それらをスタック
                displayed_image = None
                for serial in sorted(camera_images.keys()):
                    if displayed_image is None:
                        displayed_image = camera_images[serial]
                    else:
                        displayed_image = np.hstack((displayed_image, camera_images[serial]))
            
            if displayed_image is None:
                # どのカメラ画像も正常に取得できなかった場合、黒い画像を生成
                # パイプラインが設定されている場合は、その数に合わせて幅を調整
                num_cameras_expected = len(pipelines) if len(pipelines) > 0 else 1
                displayed_image = np.zeros((RESOLUTION_HEIGHT, RESOLUTION_WIDTH * num_cameras_expected, 3), dtype=np.uint8)


            # ウィンドウに記録状態を表示
            status_text = "RECORDING..." if is_recording else "READY"
            color = (0, 255, 0) if is_recording else (0, 0, 255) # 緑:記録中, 赤:待機中
            cv.putText(displayed_image, status_text, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv.LINE_AA)
            cv.imshow("RealSense Synchronization Check", displayed_image)

            key = cv.waitKey(1) & 0xFF

            # 's'キーで記録開始
            if key == ord('s'):
                if not is_recording:
                    is_recording = True
                    frame_counter = 0 # 新しい記録セッションのためにカウンタをリセット
                    print("\n--- Recording STARTED ---")
                else:
                    print("Recording is already active.")
            
            # 'e'キーで記録終了
            elif key == ord('e'):
                if is_recording:
                    is_recording = False
                    print(f"\n--- Recording STOPPED --- (Total frames recorded: {frame_counter})")
                else:
                    print("Recording is not active.")

            # フレーム記録ロジック
            # 全てのカメラからデータが取得できた場合のみ記録
            if is_recording and all(data is not None for data in current_frame_data.values()):
                for serial, data in current_frame_data.items():
                    color_img = data["color"]
                    # depth_img = data["depth"] # 深度画像は保存しないためコメントアウト
                    
                    # ファイル名にフレームカウンタとシリアル番号を含める
                    color_filename = os.path.join(RECORDING_OUTPUT_DIR, f"{serial}_frame_{frame_counter:04d}_color.png")
                    # depth_filename = os.path.join(RECORDING_OUTPUT_DIR, f"{serial}_frame_{frame_counter:04d}_depth.png") # 深度画像は保存しないためコメントアウト
                    
                    cv.imwrite(color_filename, color_img)
                    # cv.imwrite(depth_filename, depth_img) # 深度画像は保存しないためコメントアウト
                
                print(f"Recorded frame {frame_counter:04d} for all cameras.")
                frame_counter += 1

            # 'q' または 'Esc' で終了
            if key == ord('q') or key == 27:
                print("Exiting synchronization check...")
                break

    finally:
        cv.destroyAllWindows()


def main():
    serials, ctx = find_realsense_devices()
    if len(serials) < 2:
        print("ERROR: At least two RealSense devices are required for synchronization check. Exiting.")
        return

    # マスター・スレーブ設定
    if not setup_master_slave(ctx, serials):
        print("Synchronization setup failed. Exiting.")
        return

    # ストリームの有効化
    pipelines = enable_device_streams(ctx, serials)
    if not pipelines or len(pipelines) < 2:
        print("Failed to enable streams for both cameras. Exiting.")
        return

    # 同期チェックループの実行
    check_synchronization(pipelines)

    # パイプラインを停止
    for serial, pipe in pipelines.items():
        print(f"Stopping pipeline for device {serial}...")
        pipe.stop()
    print("All camera pipelines stopped.")

if __name__ == "__main__":
    main()
