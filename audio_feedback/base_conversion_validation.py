"""
base_conversion_validation.py

1台のカメラで基底変換の検証を行うスクリプト。
YOLOで検出した物体の Raw座標（カメラ基準）と User座標（頭部基準）をリアルタイム表示・CSV記録する。

操作:
  r: validation_config.json を再読み込み（カメラを止めずに設定変更可能）
  q: 終了
"""

import os
import csv
import time
import json
import numpy as np
import pyrealsense2 as rs
import cv2 as cv
from ultralytics import YOLO
from base_conversion import CameraTransformer


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_config.json")
YOLO_MODEL_PATH = "yolo_model/runs/detect/train26/weights/best.pt"


def load_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_first_camera():
    ctx = rs.context()
    for dev in ctx.query_devices():
        serial = dev.get_info(rs.camera_info.serial_number)
        print(f"[VAL] カメラ検出: {serial}")
        return serial
    return None


def create_csv_logger(step):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"validation_step{step}_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    f = open(filename, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Label", "Confidence", "Raw_X", "Raw_Y", "Raw_Z", "User_X", "User_Y", "User_Z"])
    print(f"[VAL] CSV記録開始: {filename}")
    return f, writer


def draw_info(image, config, raw, user, conf, label):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (w, 140), (30, 30, 30), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    step = config.get("step", "?")
    desc = config.get("description", "")
    offset = config.get("offset", [0, 0, 0])
    rotation = config.get("rotation", [0, 0, 0])

    cv.putText(image, f"Step {step}: {desc}", (10, 22), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv.putText(image, f"Offset: {offset}  Rotation: {rotation} deg", (10, 45), cv.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    if raw is not None:
        raw_text = f"Raw  (cam): X={raw[0]:+.3f}  Y={raw[1]:+.3f}  Z={raw[2]:+.3f} m"
        user_text = f"User(head): X={user[0]:+.3f}  Y={user[1]:+.3f}  Z={user[2]:+.3f} m"
        cv.putText(image, raw_text,  (10,  75), cv.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        cv.putText(image, user_text, (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        cv.putText(image, f"{label}  conf={conf:.2f}", (10, 125), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        cv.putText(image, "物体未検出", (10, 90), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv.putText(image, "[r]:設定再読込  [q]:終了", (w - 230, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)


def main():
    serial = find_first_camera()
    if serial is None:
        print("[VAL] エラー: カメラが見つかりません。")
        return

    config = load_config()
    transformer = CameraTransformer(config["offset"], config["rotation"])
    model = YOLO(YOLO_MODEL_PATH)

    pipeline = rs.pipeline()
    rs_config = rs.config()
    rs_config.enable_device(serial)
    rs_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    rs_config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    align = rs.align(rs.stream.color)
    pipeline.start(rs_config)

    csv_file, csv_writer = create_csv_logger(config["step"])

    print(f"[VAL] Step {config['step']}: {config['description']}")
    print(f"[VAL] Offset={config['offset']}, Rotation={config['rotation']}")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned = align.process(frames)
            color_frame = aligned.get_color_frame()
            depth_frame = aligned.get_depth_frame()
            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            results = model(color_image, verbose=False)
            annotated = results[0].plot()

            raw = user = None
            conf_val = 0.0
            label = ""

            for result in results:
                if result.boxes:
                    box = result.boxes
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf_val = float(box.conf[0])
                    label = model.names[int(box.cls[0])]
                    cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)
                    w_box, h_box = float(x2 - x1), float(y2 - y1)

                    depth_image = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units()
                    win = depth_image[
                        max(0, int(cy - h_box / 2)):int(cy + h_box / 2),
                        max(0, int(cx - w_box / 2)):int(cx + w_box / 2)
                    ]
                    win[win == 0] = np.nan
                    median_depth = float(np.nanmedian(win))

                    if not np.isnan(median_depth) and median_depth > 0:
                        intrin = depth_frame.profile.as_video_stream_profile().intrinsics
                        cam_x, cam_y, cam_z = rs.rs2_deproject_pixel_to_point(intrin, [cx, cy], median_depth)
                        raw = (cam_x, cam_y, cam_z)
                        user = transformer.transform_to_head_coords([cam_x, cam_y, cam_z])

                        ts = time.strftime("%H:%M:%S")
                        csv_writer.writerow([
                            ts, label, f"{conf_val:.3f}",
                            f"{cam_x:.4f}", f"{cam_y:.4f}", f"{cam_z:.4f}",
                            f"{user[0]:.4f}", f"{user[1]:.4f}", f"{user[2]:.4f}"
                        ])
                    break

            draw_info(annotated, config, raw, user, conf_val, label)
            cv.imshow("Validation", annotated)

            key = cv.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("r"):
                csv_file.close()
                config = load_config()
                transformer = CameraTransformer(config["offset"], config["rotation"])
                csv_file, csv_writer = create_csv_logger(config["step"])
                print(f"[VAL] 設定再読込: Step {config['step']}, Offset={config['offset']}, Rotation={config['rotation']}")

    finally:
        pipeline.stop()
        csv_file.close()
        cv.destroyAllWindows()
        print("[VAL] 終了")


if __name__ == "__main__":
    main()