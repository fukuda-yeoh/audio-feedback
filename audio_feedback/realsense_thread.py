import numpy as np
from queue import Empty, Queue
from threading import Thread

import pyrealsense2 as rs


class RealSenseThread(Thread):
    def __init__(self, serial_number=None,  *args, **kwargs):
        super(RealSenseThread, self).__init__(*args, **kwargs)

        self.stop_flag = False
        self.output_queue = Queue(maxsize=1)

        # RealSense pipeline setup
        self.pipeline = rs.pipeline()
        self.config = rs.config()

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

            if self.output_queue.full():
                try:
                    self.output_queue.get_nowait()
                except Empty:
                    pass
            self.output_queue.put((color_image, depth_frame))
        self.pipeline.stop()

    def stop(self):
        self.stop_flag = True

    def get_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
        except RuntimeError:
            return None, None

        aligned_frames = self.align.process(frames)

        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()

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

        median_depth = np.nanmedian(window)
        return median_depth

    def get_depth(self, center_pos, depth_frame):
        depth = depth_frame.get_distance(*center_pos)
        return depth

    def convert_to_3d(self, depth_frame, depth, pixel_pos):
        depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
        x, y, z = rs.rs2_deproject_pixel_to_point(depth_intrin, pixel_pos, depth)
        return x, y, z


if __name__ == "__main__":
    rs_thread = RealSenseThread()
