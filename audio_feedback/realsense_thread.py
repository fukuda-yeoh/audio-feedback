import numpy as np
from queue import Empty, Queue
from threading import Thread

import pyrealsense2 as rs


class RealSenseThread(Thread):
    def __init__(self, serial_number=None,  *args, **kwargs):
        super(RealSenseThread, self).__init__(*args, **kwargs)

        self.stop_flag = False
        self.output_queue = Queue(maxsize=100)  # YOLO用キューは最新フレームのみ

        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 【重要】修正: シリアル番号が指定されていれば、そのデバイスを有効化
        if serial_number:
            self.config.enable_device(serial_number)
            print(f"[RSThread] Enabled device with serial: {serial_number}")

        # Enable depth and color streams
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # alignモジュールの定義
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def run(self):
        # Start streaming
        self.pipeline.start(self.config)

        while not self.stop_flag:
            color_frame, depth_frame = self.get_frame()
            if color_frame is None or depth_frame is None:
                continue

            color_image = self.convert_to_array(color_frame)
            intrinsics = color_frame.profile.as_video_stream_profile().intrinsics

            self.output_queue.put(
                (
                    color_image,  # 1. NumPy配列 (YOLO入力)
                    depth_frame,  # 2. rs.frame (深度計算用)
                )
            )
        self.pipeline.stop()

    def stop(self):
        self.stop_flag = True

    def get_frame(self):
        # Wait for frames from the RealSense camera
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return None, None

        # カラーフレームにアライメントする
        aligned_frames = self.align.process(frames)

        # アライメント後のカラー画像と深度画像を取得
        color_frame = frames.get_color_frame()  # カラー画像
        depth_frame = frames.get_depth_frame()  # 深度画像

        if not color_frame or not depth_frame:
            return None, None
        else:
            return color_frame, depth_frame

    @staticmethod
    def convert_to_array(frame):
        return np.asanyarray(frame.get_data())

    def get_median_depth(self, center_pos, window_sizex, window_sizey, depth_frame):
        depth_image = self.convert_to_array(depth_frame)
        units = depth_frame.get_units()

        center_x, center_y = center_pos
        window = (
            depth_image[
                int(center_y - window_sizey / 2) : int(center_y + window_sizey / 2),
                int(center_x - window_sizex / 2) : int(center_x + window_sizex / 2),
            ]
            * units
        )
        window[window == 0] = np.nan

        # Calculate the median depth from valid points
        median_depth = np.nanmedian(window)
        return median_depth

    def get_depth(self, center_pos, depth_frame):
        depth = depth_frame.get_distance(
            *center_pos
        )  # 指定したピクセル位置の深度データを取得
        return depth

    def convert_to_3d(self, depth_frame, depth, pixel_pos):
        # ピクセル位置を3D空間の座標に変換
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, pixel_pos, depth)
        return x, y, z


if __name__ == "__main__":
    rs_thread = RealSenseThread()
