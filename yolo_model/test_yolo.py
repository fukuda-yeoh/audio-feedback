import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import numpy as np

# RealSenseのパイプライン設定
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# YOLOモデルロード（例：自分の学習済みモデル）
model = YOLO('yolo_model/runs/detect/train4/weights/best.pt')

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        
        # numpy配列に変換
        color_image = np.asanyarray(color_frame.get_data())

        # YOLO推論
        results = model(color_image)

        # 結果描画
        annotated_frame = results[0].plot()

        # ウィンドウに表示
        cv2.imshow('RealSense YOLO', annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    pipeline.stop()
    cv2.destroyAllWindows()
