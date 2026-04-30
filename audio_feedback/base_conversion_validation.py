# base_conversion_validation.py
# 基底変換の検証スクリプト（カメラ1台）
#
# 実験手順:
#   Step 1: カメラを基準位置に置いて物体を撮影 → 's'で座標記録
#   Step 2: validation_config.jsonのoffsetを[-0.2,0,0]に書き換えて
#           カメラを20cm左に移動 → 'r'で設定再読み込み → 's'で記録
#   Step 3: validation_config.jsonにrotationを追加してカメラを回転 → 'r' → 's'で記録
#
# キー操作:
#   s: 現在の座標を記録（CSVに保存）
#   r: validation_config.jsonを再読み込み（ステップ更新）
#   q: 終了

import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import json
import os
import csv
import datetime
import time
from queue import Empty

from realsense_thread import RealSenseThread
from yolo_thread import YOLOThread
from base_conversion import CameraTransformer


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_config.json")
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")


def load_validation_config():
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def find_first_camera():
    ctx = rs.context()
    devices = ctx.query_devices()
    for dev in devices:
        return dev.get_info(rs.camera_info.serial_number)
    return None


def draw_info(image, step, description, label, conf,
              raw, user, offset, rotation):
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (640, 140), (30, 30, 30), -1)
    cv.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    cv.putText(image, f"Step {step}: {description}",
               (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    cv.putText(image, f"offset={[round(v,3) for v in offset]}  rotation={rotation}",
               (10, 42), cv.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    cv.putText(image, f"[Raw]  X:{raw[0]:+.4f}  Y:{raw[1]:+.4f}  Z:{raw[2]:+.4f}",
               (10, 68), cv.FONT_HERSHEY_SIMPLEX, 0.55, (80, 80, 255), 2)
    cv.putText(image, f"[User] X:{user[0]:+.4f}  Y:{user[1]:+.4f}  Z:{user[2]:+.4f}",
               (10, 96), cv.FONT_HERSHEY_SIMPLEX, 0.55, (80, 255, 80), 2)
    cv.putText(image, f"{label} ({conf:.2f})  |  s:記録  r:設定更新  q:終了",
               (10, 122), cv.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    serial = find_first_camera()
    if serial is None:
        print("[ERROR] カメラが見つかりません")
        return

    print(f"[INFO] 使用カメラ Serial: {serial}")

    cfg = load_validation_config()
    transformer = CameraTransformer(cfg["offset"], cfg["rotation"])
    step = cfg.get("step", 1)
    description = cfg.get("description", "")

    rs_thread = RealSenseThread(serial_number=serial, name="RS-Val")
    yolo_thread = YOLOThread(input_queue=rs_thread.output_queue, name="YOLO-Val")

    yolo_thread.start()
    while not yolo_thread.ready:
        time.sleep(0.1)
    rs_thread.start()

    os.makedirs(LOG_DIR, exist_ok=True)
    log_path = os.path.join(LOG_DIR, f"validation_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    csv_file = open(log_path, mode='w', newline='', encoding='utf-8')
    writer = csv.writer(csv_file)
    writer.writerow([
        "Timestamp", "Step", "Description", "Label", "Conf",
        "Raw_X", "Raw_Y", "Raw_Z",
        "User_X", "User_Y", "User_Z",
        "Offset_X", "Offset_Y", "Offset_Z",
        "Rot_Pitch", "Rot_Yaw", "Rot_Roll"
    ])

    print(f"[Step {step}] {description}")
    print(f"  offset={cfg['offset']}  rotation={cfg['rotation']}")
    print(f"  ログ: {log_path}\n")

    last_result = None

    try:
        while True:
            try:
                color_image, depth_frame, results = yolo_thread.output_queue.get_nowait()

                if results:
                    (x1, y1, x2, y2), conf, label = results[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    w, h = x2 - x1, y2 - y1

                    median_depth = rs_thread.get_median_depth((cx, cy), w, h, depth_frame)
                    raw_x, raw_y, raw_z = rs_thread.convert_to_3d(
                        depth_frame, median_depth, (cx, cy))
                    user_x, user_y, user_z = transformer.transform_to_head_coords(
                        [raw_x, raw_y, raw_z])

                    raw = (raw_x, raw_y, raw_z)
                    user = (user_x, user_y, user_z)
                    last_result = (label, conf, raw, user)

                    cv.rectangle(color_image,
                                 (int(x1), int(y1)), (int(x2), int(y2)),
                                 (0, 255, 0), 2)
                    draw_info(color_image, step, description,
                              label, conf, raw, user,
                              cfg["offset"], cfg["rotation"])

                cv.imshow("Base Conversion Validation", color_image)

            except Empty:
                pass

            key = cv.waitKey(1)

            if key == ord('q'):
                break

            elif key == ord('s'):
                if last_result is None:
                    print("[WARN] 物体が検出されていません")
                    continue
                label, conf, raw, user = last_result
                ts = datetime.datetime.now().strftime("%H:%M:%S.%f")
                writer.writerow([
                    ts, step, description, label, f"{conf:.3f}",
                    f"{raw[0]:.5f}", f"{raw[1]:.5f}", f"{raw[2]:.5f}",
                    f"{user[0]:.5f}", f"{user[1]:.5f}", f"{user[2]:.5f}",
                    cfg["offset"][0], cfg["offset"][1], cfg["offset"][2],
                    cfg["rotation"][0], cfg["rotation"][1], cfg["rotation"][2]
                ])
                csv_file.flush()
                print(f"[Step {step}] 記録完了")
                print(f"  Raw : X={raw[0]:+.4f}  Y={raw[1]:+.4f}  Z={raw[2]:+.4f}")
                print(f"  User: X={user[0]:+.4f}  Y={user[1]:+.4f}  Z={user[2]:+.4f}")

            elif key == ord('r'):
                cfg = load_validation_config()
                transformer = CameraTransformer(cfg["offset"], cfg["rotation"])
                step = cfg.get("step", step)
                description = cfg.get("description", "")
                last_result = None
                print(f"\n[Step {step}] 設定再読み込み完了")
                print(f"  {description}")
                print(f"  offset={cfg['offset']}  rotation={cfg['rotation']}\n")

    finally:
        rs_thread.stop()
        yolo_thread.stop()
        rs_thread.join()
        yolo_thread.join()
        csv_file.close()
        cv.destroyAllWindows()
        print(f"\n[INFO] ログ保存完了: {log_path}")


if __name__ == "__main__":
    main()