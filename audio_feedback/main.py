import os

# https://github.com/opencv/opencv/issues/17687
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

import cv2 as cv

from audio_feedback.camera import CameraThread, load_intrinsic, load_extrinsic
from audio_feedback.defs import project_root
from audio_feedback.recognition import HSVColorModel, RecognitionThread
from audio_feedback.tones import SineTone

camera_no = 1

intrinsic_matrix, distortion_coeffs, fisheye = load_intrinsic(
    project_root() / "calibration" / "intrinsic.json"
)  # load undistort calibration

transformation_matrix, _, output_size = load_extrinsic(
    project_root() / "calibration" / "extrinsic.json"
)  # load perspective correction

model = HSVColorModel(
    hue_range=(100, 120), saturation_range=(144, 255), value_range=(110, 255)
)

# setup camera thread
camera_thread = CameraThread(camera_no)
if "intrinsic_matrix" in locals():
    camera_thread.set_undistort(intrinsic_matrix, distortion_coeffs, fisheye)
if "transformation_matrix" in locals():
    camera_thread.set_perspective(transformation_matrix, output_size)

# setup recognition thread
recognition_thread = RecognitionThread(model)

# setup feedback tone
tone = SineTone(
    pitch_per_second=960,
    decibels_per_second=1000,
)

# run
camera_thread.start()
recognition_thread.start()

tone.change_vol(-60)
tone.play()

while True:
    frame = camera_thread.queue.get()
    recognition_thread.in_queue.put(frame)
    result = recognition_thread.out_queue.get()

    if result:
        center = result.center

        img_width = frame.shape[0]
        lr_balance = (center[0] - (img_width / 2)) / img_width
        tone.change_lr_balance(lr_balance)
        tone.change_vol(0)
    else:
        tone.change_vol(-60)

    cv.imshow(f"Camera {camera_no}", result.annotated_img)

    if cv.waitKey(1) == ord("q"):
        recognition_thread.stop()
        recognition_thread.in_queue.put(frame)
        camera_thread.stop()
        break

cv.destroyAllWindows()
