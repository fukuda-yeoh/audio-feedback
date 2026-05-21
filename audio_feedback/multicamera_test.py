# multicamera_test.py
# 2カメラ基底変換テスト（音響なし）
# - カメラごとに独立したYOLOスレッド
# - 別ウィンドウ表示: Raw座標・User座標・ΔUser（カメラ間差分）
# - CSVログ出力

import os
import csv
import json
import time
import datetime
import numpy as np
import pyrealsense2 as rs
import cv2 as cv
from queue import Empty

from realsense_thread import RealSenseThread
from yolo_thread import YOLOThread
from base_conversion import CameraTransformer


YOLO_MODEL_PATH = "yolo_model/runs/detect/train26/weights/best.pt"


def load_camera_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "camera_config.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_realsense_serials():
    ctx = rs.context()
    return [dev.get_info(rs.camera_info.serial_number) for dev in ctx.query_devices()]


def create_csv_logger():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    filename = os.path.join(log_dir, f"multicam_test_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    f = open(filename, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(f)
    writer.writerow([
        "Timestamp", "Camera_ID", "Label", "Confidence",
        "Raw_X", "Raw_Y", "Raw_Z",
        "User_X", "User_Y", "User_Z",
        "Delta_X", "Delta_Y", "Delta_Z"
    ])
    print(f"[MAIN] CSV記録開始: {filename}")
    return f, writer


def draw_overlay(image, cam_id, raw, user, conf, label, delta):
    h, w = image.shape[:2]
    overlay = image.copy()
    cv.rectangle(overlay, (0, 0), (w, 155), (30, 30, 30), -1)
    cv.addWeighted(overlay, 0.65, image, 0.35, 0, image)

    cv.putText(image, f"Camera: {cam_id}", (10, 22),
               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    if raw is not None:
        cv.putText(image, f"Raw  (cam): X={raw[0]:+.3f}  Y={raw[1]:+.3f}  Z={raw[2]:+.3f} m",
                   (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(image, f"User(head): X={user[0]:+.3f}  Y={user[1]:+.3f}  Z={user[2]:+.3f} m",
                   (10, 75), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv.putText(image, f"{label}  conf={conf:.2f}",
                   (10, 100), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    else:
        cv.putText(image, "物体未検出", (10, 75),
                   cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    if delta is not None:
        dtext = f"Delta(L-R): X={delta[0]:+.3f}  Y={delta[1]:+.3f}  Z={delta[2]:+.3f} m"
        cv.putText(image, dtext, (10, 130),
                   cv.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1)

    cv.putText(image, "[q]:終了", (w - 100, h - 10),
               cv.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)


def main():
    connected_serials = find_realsense_serials()
    print(f"[MAIN] 接続デバイス: {connected_serials}")
    if not connected_serials:
        print("[MAIN] エラー: カメラが見つかりません。")
        return

    camera_config = load_camera_config()
    camera_systems = {}

    for serial in connected_serials:
        if serial not in camera_config:
            print(f"[MAIN] 警告: 未登録カメラ (Serial: {serial}) はスキップします。")
            continue

        settings = camera_config[serial]
        cam_id = settings["id"]
        print(f"[MAIN] セットアップ中: {cam_id} Camera (Serial: {serial})")

        transformer = CameraTransformer(settings["offset"], settings["rotation"])
        rs_thread = RealSenseThread(serial_number=serial, name=f"RS-{cam_id}")
        yolo_thread = YOLOThread(input_queue=rs_thread.output_queue, name=f"YOLO-{cam_id}")

        camera_systems[serial] = {
            "id": cam_id,
            "rs_thread": rs_thread,
            "yolo_thread": yolo_thread,
            "transformer": transformer,
            "latest_user": None,
        }

        yolo_thread.start()
        while not yolo_thread.ready:
            time.sleep(0.05)
        rs_thread.start()
        print(f"[MAIN] {cam_id} Camera 起動完了")

    if not camera_systems:
        print("[MAIN] エラー: 設定済みカメラが見つかりません。camera_config.json を確認してください。")
        return

    csv_file, csv_writer = create_csv_logger()
    print("\n[MAIN] システム稼働開始。[q]:終了")

    try:
        while True:
            for serial, system in camera_systems.items():
                cam_id = system["id"]
                transformer = system["transformer"]
                rs_thread = system["rs_thread"]
                yolo_queue = system["yolo_thread"].output_queue

                try:
                    annotated_frame, depth_frame, results = yolo_queue.get_nowait()
                except Empty:
                    continue
                except Exception as e:
                    print(f"[{cam_id}] キュー取得エラー: {e}")
                    continue

                raw = user = None
                conf_val = 0.0
                label = ""

                if results and depth_frame is not None:
                    (x1, y1, x2, y2), conf_val, label = results[0]
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    w_box = float(x2 - x1)
                    h_box = float(y2 - y1)

                    try:
                        median_depth = rs_thread.get_median_depth(
                            (cx, cy), w_box, h_box, depth_frame
                        )
                        if not np.isnan(median_depth) and median_depth > 0:
                            cam_x, cam_y, cam_z = rs_thread.convert_to_3d(
                                depth_frame, median_depth, [cx, cy]
                            )
                            raw = (cam_x, cam_y, cam_z)
                            user_arr = transformer.transform_to_head_coords([cam_x, cam_y, cam_z])
                            user = tuple(user_arr)
                            system["latest_user"] = user
                    except Exception as e:
                        print(f"[{cam_id}] 深度変換エラー: {e}")

                # ΔUser: 全カメラのlatest_userが揃っている場合のみ計算
                all_users = {s: sys["latest_user"] for s, sys in camera_systems.items()
                             if sys["latest_user"] is not None}
                delta = None
                if len(all_users) == 2:
                    user_vals = list(all_users.values())
                    delta = tuple(user_vals[0][i] - user_vals[1][i] for i in range(3))

                draw_overlay(annotated_frame, cam_id, raw, user, conf_val, label, delta)
                cv.imshow(f"Camera: {cam_id}", annotated_frame)

                if raw is not None:
                    ts = datetime.datetime.now().strftime("%H:%M:%S.%f")
                    d = delta if delta else ("", "", "")
                    csv_writer.writerow([
                        ts, cam_id, label, f"{conf_val:.3f}",
                        f"{raw[0]:.4f}", f"{raw[1]:.4f}", f"{raw[2]:.4f}",
                        f"{user[0]:.4f}", f"{user[1]:.4f}", f"{user[2]:.4f}",
                        f"{d[0]:.4f}" if d[0] != "" else "",
                        f"{d[1]:.4f}" if d[1] != "" else "",
                        f"{d[2]:.4f}" if d[2] != "" else "",
                    ])
                    print(
                        f"[{cam_id}] {label}: "
                        f"Raw({raw[0]:+.2f},{raw[1]:+.2f},{raw[2]:+.2f}) "
                        f"User({user[0]:+.2f},{user[1]:+.2f},{user[2]:+.2f})"
                        + (f"  ΔUser({delta[0]:+.3f},{delta[1]:+.3f},{delta[2]:+.3f})" if delta else ""),
                        end="\r"
                    )

            if cv.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        print("\n[MAIN] 終了処理中...")
        csv_file.close()
        print("[MAIN] CSV保存完了")
        for system in camera_systems.values():
            system["rs_thread"].stop()
            system["yolo_thread"].stop()
        for system in camera_systems.values():
            system["rs_thread"].join()
            system["yolo_thread"].join()
        cv.destroyAllWindows()
        print("[MAIN] 終了")


if __name__ == "__main__":
    main()
