# yolo_thread.py

import threading
from queue import Queue, Empty
import time
import numpy as np
from ultralytics import YOLO
import cv2 as cv


class YOLOThread(threading.Thread):
    """
    YOLO推論を実行し、検出結果（3D座標）を共有データに格納するスレッド。
    """

    def __init__(self, input_queue, *args, **kwargs):
        super().__init__(*args, **kwargs) # 親クラスに引数を渡す
        
        self.model = YOLO("yolo_model/runs/detect/train26/weights/best.pt")
        print(self.model.names)

        self.input_queue = input_queue
        print(self.model.names)

        self.input_queue = input_queue
# yolo_thread.py

import threading
from queue import Queue, Empty
import time
import numpy as np
from ultralytics import YOLO
import cv2 as cv


class YOLOThread(threading.Thread):
    """
    YOLO推論を実行し、検出結果（3D座標）を共有データに格納するスレッド。
    """

    def __init__(self, input_queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.model = YOLO("yolo_model/runs/detect/train26/weights/best.pt")
        print(self.model.names)

        self.input_queue = input_queue
        self.output_queue = Queue(maxsize=1)
        self._stop_event = threading.Event()

        self.raw_frame = None
        self.annotated_frame = None
        self.depth_frame = None
        self.detection_result = None
        self.ready = False

    def run(self):
        while not self._stop_event.is_set():
            self.ready = True
            try:
                color_image, depth_frame = self.input_queue.get()

                # --- YOLO 推論 ---
                results = self.model(color_image, verbose=False)
                annotated_frame = results[0].plot()

                my_results = []

                for result in results:
                    if result.boxes:
                        box = result.boxes

                        x1, y1, x2, y2 = box.xyxy[0]
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.model.names[cls]

                        my_results.append(((x1, y1, x2, y2), conf, label))

                self.raw_frame = color_image
                self.depth_frame = depth_frame
                self.annotated_frame = annotated_frame

                if self.output_queue.full():
                    try:
                        self.output_queue.get_nowait()
                    except Empty:
                        pass
                self.output_queue.put((annotated_frame, depth_frame, my_results))

            except Exception as e:
                print(f"[{self.name}] Error in processing loop: {e}")
                time.sleep(0.1)

        print(f"[{self.name}] Stopped.")

    def stop(self):
        self._stop_event.set()
        self.output_queue = Queue()
        self._stop_event = threading.Event()

        self.raw_frame = None
        self.annotated_frame = None
        self.depth_frame = None
        self.detection_result = None
        self.ready = False

    def run(self):
        while not self._stop_event.is_set():
            self.ready = True
            try:
                # --- 修正点: 3つの要素を期待するように変更 ---
                # (color_image, depth_frame) を取得
                color_image, depth_frame = self.input_queue.get()

                # --- YOLO 推論 ---
                results = self.model(color_image, verbose=False)
                annotated_frame = results[0].plot()

                my_results = []

                # # 単一カメラ、単一物体（最も信頼度の高いもの）を想定
                for result in results:
                    if result.boxes:
                        box = result.boxes  # 最も信頼度の高いボックス

                        x1, y1, x2, y2 = box.xyxy[0]
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        label = self.model.names[cls]

                        cx = int((x1 + x2) / 2)
                        cy = int((y1 + y2) / 2)

                        my_results.append(((x1, y1, x2, y2), conf, label))

                        # 描画
                        cv.rectangle(
                            self.annotated_frame,
                            (int(x1), int(y1)),
                            (int(x2), int(y2)),
                            (0, 255, 0),
                            2,
                        )
                        text = (
                            f"{label} | Confidence: {conf:.2f} )"
                        )
                        cv.putText(
                            self.annotated_frame,
                            text,
                            (int(x1), int(y1) - 10),
                            cv.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (255, 255, 255),
                            1,
                        )

                self.raw_frame = color_image
                self.depth_frame = depth_frame
                self.annotated_frame = annotated_frame

                self.output_queue.put((annotated_frame, depth_frame, my_results))

            except Exception as e:
                print(f"[{self.name}] Error in processing loop: {e}")
                time.sleep(0.1)

        print(f"[{self.name}] Stopped.")

    def stop(self):
        self._stop_event.set()
