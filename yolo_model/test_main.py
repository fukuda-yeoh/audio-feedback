from ultralytics import YOLO
import torch

# Load a model
model = YOLO("yolo11n.pt")  # load an official model
model = YOLO("runs/detect/train4/weights/best.pt")  # load a custom model

# # Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image

# from ultralytics import YOLO
# import cv2

# # モデルの読み込み（学習済みモデルに置き換えてOK）
# model = YOLO("runs/detect/train/weights/best.pt")  # 例: "runs/detect/train/weights/best.pt"

# # カメラを起動（0はPC内蔵カメラ、外部なら1など）
# cap = cv2.VideoCapture(1)

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # YOLOv8で推論（BGR -> RGB変換は不要、OpenCVでそのまま可）
#     results = model.predict(frame, conf=0.3, stream=True)  # stream=Trueで高速処理

#     # 結果を描画
#     for r in results:
#         annotated_frame = r.plot()  # 検出結果を描画したフレームを取得

#         # 表示
#         cv2.imshow("YOLO Detection", annotated_frame)

#     # qキーで終了
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break

# # 後処理
# cap.release()
# cv2.destroyAllWindows()