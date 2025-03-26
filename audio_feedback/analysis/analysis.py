import cv2
import numpy as np
import math
import csv
from pathlib import Path


def get_object_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        M = cv2.moments(largest_contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return (cx, cy)

# HSV色範囲を設定
lower_light_blue = np.array([100, 90, 40])  # 青色の範囲
upper_light_blue = np.array([115, 150, 120])

# lower_light_blue = np.array([90, 60, 36])  # 青色の範囲q
# upper_light_blue = np.array([115, 160, 70])


lower_orange = np.array([15, 160, 100])
upper_orange = np.array([25, 255, 255])

# スケールファクター設定 (ピクセルからセンチメートルに変換)
scale_factor = 0.311  # 例: 1ピクセル = 0.1センチメートル

# 動画キャプチャ
video_dir = Path(r"C:\Users\oobuh\卓球\audio-feedback\analaysis")
for f in video_dir.glob("*.mp4"):
    p_id, condition, trial = f.stem.split("-")
    cap = cv2.VideoCapture(str(f))  # または動画ファイルのパス

    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    # 許容範囲 (ピクセル単位で設定)
    tolerance = 10

    # 距離データを保存するファイルの準備
    output_file = "distances.csv"
    if not Path(output_file).is_file():
        with open(output_file, "w", encoding="utf-8", newline="") as f:
            csv_writer = csv.writer(f, delimiter=",")
            csv_writer.writerow(["p_id", "条件", "回数", "frame", "distance"])

    # フレームレートを取得
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # 青色とオレンジ色のマスクを作成
        mask_light_blue = cv2.inRange(hsv, lower_light_blue, upper_light_blue)
        mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

        center_blue = get_object_center(mask_light_blue)
        center_orange = get_object_center(mask_orange)

        if center_blue:
            cv2.circle(frame, center_blue, 10, (255, 0, 0), -1)  # 青色の物体
        if center_orange:
            cv2.circle(frame, center_orange, 10, (0, 165, 255), -1)  # オレンジ色の物体

        if center_blue and center_orange:
            # ピクセル距離を計算
            pixel_distance = math.sqrt(
                (center_blue[0] - center_orange[0]) ** 2
                + (center_blue[1] - center_orange[1]) ** 2
            )

            # ピクセル距離をセンチメートルに変換
            distance_cm = pixel_distance * scale_factor

            distance_str = f"Distance: {distance_cm:.2f} cm"
            cv2.putText(
                frame,
                distance_str,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

            # オレンジの物体の位置が x=0 近くの場合、距離を保存
            # if abs(center_orange[0]) < tolerance:
            with open(output_file, "a", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([p_id, condition, trial, frame_count, distance_cm])

        cv2.imshow("frame", frame)

        # フレームレートに基づいた待機時間を設定
        wait_time = int(1000 / fps)  # ms
        if cv2.waitKey(wait_time) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
