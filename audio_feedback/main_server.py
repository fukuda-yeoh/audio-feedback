import threading
from queue import Queue

import cv2 as cv
from flask import Response, Flask, render_template

from audio_feedback.camera import CameraThread, load_intrinsic, load_extrinsic
from audio_feedback.defs import project_root
from audio_feedback.recognition import HSVColorModel, RecognitionThread
from audio_feedback.tones import SineTone

app = Flask(__name__)
ip = "0.0.0.0"
port = 8080
camera_no = 1

stream_queue = Queue()


@app.route("/")
def index():
    # return the rendered template
    return render_template("index.html")


@app.route("/video_feed")
def video_feed():
    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def generate():
    global stream_queue

    while True:
        steam_frame = stream_queue.get(block=True)
        (flag, encoded_image) = cv.imencode(".jpg", steam_frame)
        if not flag:
            continue
        # yield the output frame in the byte format
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + bytearray(encoded_image) + b"\r\n"
        )


if __name__ == "__main__":
    intrinsic_matrix, distortion_coeffs, fisheye = load_intrinsic(
        project_root() / "calibration" / "intrinsic.json"
    )  # load undistort calibration

    transformation_matrix, _, output_size = load_extrinsic(
        project_root() / "calibration" / "extrinsic.json"
    )  # load perspective correction

    model = HSVColorModel(
        hue_range=(100, 115), saturation_range=(25, 255), value_range=(150, 255)
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

    threading.Thread(
        target=lambda: app.run(host=ip, port=port, debug=True, use_reloader=False)
    ).start()

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

        stream_queue.put(frame)
