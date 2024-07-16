from queue import Queue
from threading import Thread

import cv2 as cv

from audio_feedback.camera.extrinsic_calc import change_perspective
from audio_feedback.camera.intrinsic_calc import get_undistort_funcs, undistort_map


class CameraThread(Thread):
    def __init__(self, camera_no=0, *args, **kwargs):
        super(CameraThread, self).__init__(*args, **kwargs)

        self.camera_no = camera_no
        self.queue = Queue(maxsize=30)

        self.intrinsic_matrix, self.distortion_coeffs, self.fisheye, self.scale = (
            None,
            None,
            None,
            None,
        )
        self.transformation_matrix, self.output_size = None, None

        self.stop_flag = False
        self.cap = None
        self.frame = None

    def run(self):
        self.cap = cv.VideoCapture(self.camera_no)
        ret, frame = self.cap.read()
        if ret:
            if self.distortion_coeffs is not None and self.intrinsic_matrix is not None:
                map_x, map_y = get_undistort_funcs(
                    frame.shape,
                    self.intrinsic_matrix,
                    self.distortion_coeffs,
                    self.fisheye,
                    self.scale,
                )  # setup mapping
            else:
                map_x, map_y = None, None
        else:
            raise "Could not read from camera"

        while not self.stop_flag:
            ret, frame = self.cap.read()
            if ret:
                if map_x is not None or map_y is not None:
                    frame = undistort_map(frame, map_x, map_y)  # undistort
                if self.transformation_matrix is not None:
                    frame = change_perspective(
                        frame, self.transformation_matrix, self.output_size
                    )
                self.queue.put(frame)
        self.cap.release()

    def set_undistort(
        self, intrinsic_matrix, distortion_coeffs, fisheye=False, scale=1.0
    ):
        self.intrinsic_matrix, self.distortion_coeffs, self.fisheye, self.scale = (
            intrinsic_matrix,
            distortion_coeffs,
            fisheye,
            scale,
        )

    def set_perspective(self, transformation_matrix, output_size):
        self.transformation_matrix, self.output_size = (
            transformation_matrix,
            output_size,
        )

    def stop(self):
        self.stop_flag = True


if __name__ == "__main__":

    cam_no = 1
    camera_thread = CameraThread(cam_no)
    camera_thread.start()

    while True:
        raw_frame = camera_thread.queue.get()

        cv.imshow(f"Camera {cam_no}", raw_frame)
        if cv.waitKey(1) == ord("q"):
            camera_thread.stop()
            break

    camera_thread.stop()
