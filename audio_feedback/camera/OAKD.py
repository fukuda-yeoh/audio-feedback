import depthai as dai
import numpy as np
from calc import HostSpatialsCalc
from utility import TextHelper
import queue
from threading import Thread


class OAKDThread(Thread):
    def __init__(self, *args, **kwargs):
        super(OAKDThread, self).__init__(*args, **kwargs)
        self.pipeline = dai.Pipeline()

        # RGBカメラの設定
        self.color_camera = self.pipeline.create(dai.node.ColorCamera)
        self.color_camera.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.color_camera.setResolution(
            dai.ColorCameraProperties.SensorResolution.THE_800_P
        )
        self.color_camera.setFps(30)

        self.xoutColor = self.pipeline.create(dai.node.XLinkOut)
        self.xoutColor.setStreamName("color")
        self.color_camera.video.link(self.xoutColor.input)

        # Set up stereo camera
        self.monoLeft = self.pipeline.create(dai.node.MonoCamera)
        self.monoRight = self.pipeline.create(dai.node.MonoCamera)
        self.stereo = self.pipeline.create(dai.node.StereoDepth)

        self.monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        self.monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        self.monoRight.setResolution(
            dai.MonoCameraProperties.SensorResolution.THE_800_P
        )
        self.monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(False)
        self.monoLeft.out.link(self.stereo.left)
        self.monoRight.out.link(self.stereo.right)

        self.xoutDepth = self.pipeline.create(dai.node.XLinkOut)
        self.xoutDepth.setStreamName("depth")
        self.stereo.depth.link(self.xoutDepth.input)

        # デバイスの作成
        self.device = dai.Device(self.pipeline)
        self.colorQueue = self.device.getOutputQueue(
            name="color", maxSize=4, blocking=False
        )
        self.depthQueue = self.device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        # 深度計算
        self.spatial_calc = HostSpatialsCalc(device=self.device)
        # 出力用
        self.output_queue = queue.Queue()

        self.stop_flag = False

    # メイン
    def run(self):
        while not self.stop_flag:
            depthData = self.depthQueue.get()
            colorData = self.colorQueue.get()
            frame = colorData.getCvFrame()
            self.output_queue.put(
                (
                    colorData,
                    depthData,
                    frame,
                )
            )
        self.device.close()

    def stop(self):
        self.stop_flag = True

    def get_colorframe(self):
        # RGB映像を取得
        colorData = self.colorQueue.get()
        frame = colorData.getCvFrame()
        return frame

    def set_delta_roi(self, delta):
        self.spatial_calc.setDeltaRoi(delta)

    # 深度計算
    def get_spatial_coords(self, depth_frame, center):
        spatials, _ = self.spatial_calc.calc_spatials(
            depth_frame, center, averaging_method=np.median
        )
        return spatials["x"] / 1000, spatials["y"] / 1000, spatials["z"] / 1000
