import os
import cv2 as cv
import cv2
import numpy as np
import depthai as dai

from audio_feedback.camera.OAKD import OAKDThread
from audio_feedback.recognition import HSVColorModel, RecognitionThread

# OAK-Dを開始する
oakd_thread = OAKDThread()

# モデルのHSVの設定（赤色検出）
model = HSVColorModel(
    hue_range=(0, 10), saturation_range=(100, 255), value_range=(100, 255)
)

# 物体の認識やトラッキングを行うためのスレッドを生成するクラス
recognition_thread = RecognitionThread(model, oakd_thread.output_queue)

recognition_thread.start()
oakd_thread.start()

try:
    while True:
        # 物体認識の結果処理
        color_frame, depth_frame, result = recognition_thread.out_queue.get()
        if color_frame is None:
            continue

        color_image = color_frame.getCvFrame()

        if result:
            center = result.center
            x, y, z = oakd_thread.get_spatial_coords(depth_frame, center)

            # 画像に3D座標を注釈として表示
            center_x, center_y = int(center[0]), int(center[1])
            cv.putText(
                color_image,
                f"Depth: ({x:.2f}, {y:.2f}, {z:.2f})",
                (center_x, center_y - 20),
                cv.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                2,
            )

            # 検出物体にマーカー
            cv.circle(color_image, (center_x, center_y), 10, (0, 255, 0), 2)

        # 結果の映像を表示
        cv.imshow("Color Frame", color_image)

        if cv.waitKey(1) & 0xFF == ord("q"):
            recognition_thread.stop()
            oakd_thread.stop()
            break
finally:
    cv.destroyAllWindows()
