# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO

# --- 追加: 音響フィードバック関連のインポート ---
import synthizer
import os
# from audio_feedback.defs import project_root # 以前のコードに基づき、今回はパスを直接指定または定義
# 注意: project_root()が未定義のため、ここでは仮のパスで代用します
# 実際のプロジェクト構成に合わせてパスを修正してください
SOUND_FILE_PATH = "sound_files/1000Hz_v2.wav" 
# ---------------------------------------------


# YOLOモデルのロード（パスを適切に設定してください）
try:
    yolo_model = YOLO('yolo_model/runs/detect/train4/weights/best.pt')
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'runs/detect/train4/weights/best.pt' is the correct path to your YOLO model.")
    exit()


# --- 追加: Synthizerの初期化と設定関数 ---
def setup_synthizer():
    synthizer.initialize()
    context = synthizer.Context()
    context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
    context.default_distance_model.value = synthizer.DistanceModel.EXPONENTIAL

    # 仮のパスを使用。実際の環境に合わせて修正してください
    if not os.path.exists(SOUND_FILE_PATH):
        print(f"Error: Sound file not found at {SOUND_FILE_PATH}")
        synthizer.shutdown()
        exit()

    buffer = synthizer.Buffer.from_file(SOUND_FILE_PATH)
    generator = synthizer.BufferGenerator(context)
    generator.gain.value = 1
    generator.pitch_bend.value = 1
    generator.buffer.value = buffer
    generator.looping.value = True

    source = synthizer.Source3D(context)
    source.add_generator(generator)
    source.distance_model.value = synthizer.DistanceModel.EXPONENTIAL
    source.rolloff.value = 1.0
    source.distance_ref.value = 0.1
    source.distance_max.value = 6.0
    
    # 初めは一時停止状態で、検出時に再生する
    # source.play() # playは検出時に行う
    return source, generator, context, True # 最後のTrueはsound_onフラグ
# ---------------------------------------------


def findDevices():
    ctx = rs.context()
    serials = []
    if len(ctx.devices) > 0:
        for dev in ctx.devices:
            print(
                "Found device: ",
                dev.get_info(rs.camera_info.name),
                " ",
                dev.get_info(rs.camera_info.serial_number),
            )
            serials.append(dev.get_info(rs.camera_info.serial_number))
    else:
        print("No Intel Device connected")
    return serials, ctx


def setupMasterSlave(serials, ctx):
    if len(serials) < 2:
        print("At least two devices are required for master-slave setup. Synchronization will not be effective.")
        return

    for i, device in enumerate(ctx.devices):
        print(
            f"Configuring device {i+1} with serial {device.get_info(rs.camera_info.serial_number)}"
        )
        try:
            if i == 0:
                print(
                    f"Setting device {device.get_info(rs.camera_info.serial_number)} as master."
                )
                device.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 1)
            else:
                print(
                    f"Setting device {device.get_info(rs.camera_info.serial_number)} as slave."
                )
                device.first_depth_sensor().set_option(rs.option.inter_cam_sync_mode, 2)
        except Exception as e:
            print(f"Could not set sync mode for device {device.get_info(rs.camera_info.serial_number)}: {e}")
            print("Ensure the device has a depth sensor if you intend to use sync mode.")


def enableDevices(
    serials, ctx, resolution_width=640, resolution_height=480, frame_rate=30
):
    pipelines = []
    # RealSenseカメラのキャリブレーション情報を保持する辞書
    intrinsics_map = {} 
    
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        
        # 深度ストリームの有効化
        cfg.enable_stream(
            rs.stream.depth,
            resolution_width,
            resolution_height,
            rs.format.z16, # 深度データはz16フォーマット
            frame_rate,
        )
        # カラーストリームの有効化
        profile = cfg.enable_stream(
            rs.stream.color,
            resolution_width,
            resolution_height,
            rs.format.bgr8,
            frame_rate,
        )
        
        pipe_profile = pipe.start(cfg)
        
        # カメラの内部パラメータを取得 (3D座標計算に必要)
        color_stream = pipe_profile.get_stream(rs.stream.color)
        intrinsics_map[serial] = color_stream.as_video_stream_profile().get_intrinsics()
        
        pipelines.append([serial, pipe])

    time.sleep(1)

    # パイプラインと対応するカメラ内部パラメータの辞書を返す
    return pipelines, intrinsics_map


# --- RealSenseのキャリブレーション情報を使ってピクセル座標を3D座標に変換するヘルパー関数 ---
def get_spatial_coords(depth_frame, center_pixel, depth_intrin):
    # ピクセル座標 (u, v)
    u, v = int(center_pixel[0]), int(center_pixel[1])
    
    # 深度フレームから深度値を取得 (単位は mm)
    depth_scale = 0.001 # RealSenseの深度値はmm、メートルに変換
    depth_value = depth_frame.as_depth_frame().get_distance(u, v) 
    
    if depth_value == 0:
        return None # 深度が無効な場合はNoneを返す
        
    # RealSense SDKの関数を使用して、2Dピクセル座標と深度値から3D実世界座標 (x, y, z) を計算
    # 単位はメートル
    point_in_meters = rs.rs2_deproject_pixel_to_point(depth_intrin, [u, v], depth_value)
    
    # RealSenseの座標系 (X:右, Y:下, Z:前) を使用
    return point_in_meters[0], point_in_meters[1], point_in_meters[2]


def VisualizeAndDetect(pipelines, intrinsics_map, model, source, source_sound_on_ref):
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    all_annotated_frames = []
    found_object = False # このフレームでオブジェクトが見つかったかどうかのフラグ
    
    # デバイスが複数あっても、ここでは最初のデバイスの検出結果のみを音響フィードバックに使用
    # どのデバイスの結果を使うかを決定してください (例: `pipelines[0]`)

    for serial, pipe in pipelines:
        try:
            frames = pipe.wait_for_frames(5000)

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame() # 深度フレームを取得

            if not color_frame or not depth_frame:
                print(f"No valid color or depth frame from device {serial}. Skipping.")
                continue

            color_image = np.asanyarray(color_frame.get_data())

            # YOLO検出
            results = model(color_image, verbose=False)
            annotated_frame = results[0].plot()
            
            # --- 検出結果の処理と音響フィードバック ---
            if results[0].boxes and serial == pipelines[0][0]: # 最初のカメラでのみ処理
                
                # 検出されたオブジェクトの中心座標を取得 (ここでは最初の検出結果を使用)
                box = results[0].boxes.xyxy[0].cpu().numpy().astype(int)
                center_x = int((box[0] + box[2]) / 2)
                center_y = int((box[1] + box[3]) / 2)
                
                center_pixel = (center_x, center_y)
                
                # 3D座標の計算
                intrinsics = intrinsics_map[serial]
                x, y, z = get_spatial_coords(depth_frame, center_pixel, intrinsics)
                
                if x is not None:
                    found_object = True
                    
                    # RealSense座標系 (X:右, Y:下, Z:前) -> Synthizer座標系 (X:右, Y:上, Z:奥/前)
                    # Synthizer: X:右, Y:上, Z:奥 (正の値が手前、負の値が奥/RealSenseと逆)
                    # 変換: x -> x, y -> -y (RealSenseのY軸が下向きのため), z -> -z (RealSenseのZ軸が前向きのため)
                    ball_position = (x, -y, -z)
                    source.position.value = ball_position

                    # 画像に3D座標を注釈として表示
                    cv.putText(
                        annotated_frame,
                        f"Position: ({x:.2f}, {y:.2f}, {z:.2f}) m",
                        (center_x, center_y - 20),
                        cv.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 255),
                        2,
                    )

                    # 音源の再生
                    if not source_sound_on_ref[0]:
                        source.play()
                        source_sound_on_ref[0] = True
            
            all_annotated_frames.append(annotated_frame)
            # ---------------------------------------------

        except RuntimeError as e:
            print(f"RuntimeError for device {serial}: {e}. Skipping frame.")
            continue
        except Exception as e:
            print(f"An unexpected error occurred for device {serial}: {e}. Skipping frame.")
            continue
            
    # --- 全体の検出結果に基づく音源の一時停止 ---
    if not found_object and source_sound_on_ref[0]:
        source.pause()
        source_sound_on_ref[0] = False

    # すべてのカメラからのフレームが収集された後、結合して表示
    if all_annotated_frames:
        combined_frame = np.hstack(all_annotated_frames)
        cv.imshow("RealSense YOLO + Synthizer - Combined View", combined_frame)
    else:
        print("No frames received from any device to display.")

    key = cv.waitKey(1)
    if key & 0xFF == ord("q") or key == 27:
        cv.destroyAllWindows()
        return True

    return False


def pipelineStop(pipelines):
    for serial, pipe in pipelines:
        print(f"Stopping pipeline for device {serial}...")
        pipe.stop()
    print("All pipelines stopped.")


# -------Main program--------

# 1. Synthizerのセットアップ
synthizer_source, _, synthizer_context, source_sound_on = setup_synthizer()
# 可変フラグとしてリストを使用
source_sound_on_ref = [source_sound_on] 


# 2. RealSenseデバイスのセットアップ
serials, ctx = findDevices()

if len(serials) < 1:
    print("FATAL ERROR: No RealSense devices found. Exiting.")
    synthizer.shutdown()
    exit()

if len(serials) < 2:
    print("WARNING: Less than two devices found. Master-slave synchronization will not be effective. Proceeding with available devices.")

setupMasterSlave(serials, ctx)

resolution_width = 640
resolution_height = 480
frame_rate = 30

# 3. RealSenseデバイスの有効化とキャリブレーション情報の取得
pipelines, intrinsics_map = enableDevices(serials, ctx, resolution_width, resolution_height, frame_rate)

try:
    while True:
        # 4. 検出、可視化、音響フィードバックの実行
        exit_program = VisualizeAndDetect(pipelines, intrinsics_map, yolo_model, synthizer_source, source_sound_on_ref)
        if exit_program:
            print("Program closing...")
            break
finally:
    # 5. 終了処理
    pipelineStop(pipelines)
    cv.destroyAllWindows()
    # --- 追加: Synthizerのシャットダウン ---
    synthizer.shutdown()
    # -------------------------------------