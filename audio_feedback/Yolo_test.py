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
    try:
        align_to = rs.stream.color
        align = rs.align(align_to)

        while True:
            frames_combined = []

            # ✅ 修正：serial, pipe のペアを扱う
            for i, (serial, pipe) in enumerate(pipelines):
                frames = pipe.wait_for_frames()
                aligned_frames = align.process(frames)

                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())

                # YOLOによる物体検出
                results = model(color_image, verbose=False)
                boxes = results[0].boxes

                # カメラインストリンシクス
                intr = color_frame.profile.as_video_stream_profile().intrinsics

                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]

                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    depth = depth_frame.get_distance(cx, cy)

                    if depth == 0:
                        continue

                    point_3d = rs.rs2_deproject_pixel_to_point(intr, [cx, cy], depth)
                    x, y, z = point_3d

                    # 描画
                    cv.rectangle(color_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    text = f"{conf:.2f} | D={depth:.2f}m | ({x:.2f},{y:.2f},{z:.2f})"
                    cv.putText(color_image, text, (int(x1), int(y1) - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                frames_combined.append(color_image)

            if len(frames_combined) > 0:
                combined = np.hstack(frames_combined)
                cv.imshow('YOLO + Depth 3D Position', combined)

            key = cv.waitKey(1)
            if key == ord('q') or key == 27:
                break

    except Exception as e:
        print("Error in VisualizeAndDetect:", e)



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