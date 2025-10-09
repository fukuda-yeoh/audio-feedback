# Libraries
import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO # YOLOモデルのためにultralyticsをインポート

# YOLOモデルのロード（パスを適切に設定してください）
try:
    yolo_model = YOLO('yolo_model/runs/detect/train4/weights/best.pt')
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure 'runs/detect/train4/weights/best.pt' is the correct path to your YOLO model.")
    exit()


def findDevices():
    ctx = rs.context()  # Create librealsense context for managing devices
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
    for serial in serials:
        pipe = rs.pipeline(ctx)
        cfg = rs.config()
        cfg.enable_device(serial)
        
        cfg.enable_stream(
            rs.stream.depth,
            resolution_width,
            resolution_height,
            rs.format.z16, # 深度データはz16フォーマット
            frame_rate,
        )
        cfg.enable_stream(
            rs.stream.color,
            resolution_width,
            resolution_height,
            rs.format.bgr8,
            frame_rate,
        )
        pipe.start(cfg)
        pipelines.append([serial, pipe])

    time.sleep(1) # カメラ起動のための短い待機時間

    return pipelines


def VisualizeAndDetect(pipelines, model):
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    # すべてのカメラからの処理済みフレームを格納するリスト
    all_annotated_frames = []

    for serial, pipe in pipelines:
        try:
            frames = pipe.wait_for_frames(5000)

            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()

            if not color_frame:
                print(f"No color frame from device {serial}. Skipping.") # デバッグ用ログ
                continue

            color_image = np.asanyarray(color_frame.get_data())

            results = model(color_image, verbose=False)

            annotated_frame = results[0].plot()
            all_annotated_frames.append(annotated_frame)

        except RuntimeError as e:
            print(f"RuntimeError for device {serial}: {e}. Skipping frame.") # デバッグ用ログ
            continue
        except Exception as e:
            print(f"An unexpected error occurred for device {serial}: {e}. Skipping frame.")
            continue

    # すべてのカメラからのフレームが収集された後、結合して表示
    if all_annotated_frames:
        # フレームを水平方向に結合
        combined_frame = np.hstack(all_annotated_frames)
        cv.imshow("RealSense YOLO - Combined View", combined_frame)
    else:
        print("No frames received from any device to display.") # 表示するフレームがない場合

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

serials, ctx = findDevices()

if len(serials) < 2:
    print("WARNING: Less than two devices found. Master-slave synchronization will not be effective. Proceeding with available devices.")

setupMasterSlave(serials, ctx)

resolution_width = 640
resolution_height = 480
frame_rate = 30

pipelines = enableDevices(serials, ctx, resolution_width, resolution_height, frame_rate)

try:
    while True:
        exit_program = VisualizeAndDetect(pipelines, yolo_model)
        if exit_program:
            print("Program closing...")
            break
finally:
    pipelineStop(pipelines)
    cv.destroyAllWindows()