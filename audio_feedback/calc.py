#caclファイル
import math
import numpy as np
import depthai as dai

#OADKカメラの深度計算
class HostSpatialsCalc:
    # We need device object to get calibration data
    def __init__(self, device):
        self.calibData = device.readCalibration()

        # Values
        self.DELTA = 5 #ROIの幅は、選択した領域の広さを表すし、深度データの計算範囲に影響
        self.THRESH_LOW = 200 # 20cm
        self.THRESH_HIGH = 30000 # 30m

    # 深度情報の下限 (THRESH_LOW)・上限 (THRESH_HIGH) を動的に変更できる
    def setLowerThreshold(self, threshold_low):
        self.THRESH_LOW = threshold_low
    def setUpperThreshold(self, threshold_low):
        self.THRESH_HIGH = threshold_low

    # ROI のサイズ (DELTA) を変更し、深度の平均化に使うピクセル数を調整する
    def setDeltaRoi(self, delta):
        self.DELTA = delta

    #
    def _check_input(self, roi, frame): # Check if input is ROI or point. If point, convert to ROI
        if len(roi) == 4: return roi
        if len(roi) != 2: raise ValueError("You have to pass either ROI (4 values) or point (2 values)!")

        # Limit the point so ROI won't be outside the frame
        self.DELTA = 5 # Take 10x10 depth pixels around point for depth averaging
        x = int(min(max(roi[0], self.DELTA), frame.shape[1] - self.DELTA))
        y = int(min(max(roi[1], self.DELTA), frame.shape[0] - self.DELTA))
        return (x-self.DELTA,y-self.DELTA,x+self.DELTA,y+self.DELTA)
    
    # 物体の位置 offset を基準に、カメラの視野角 HFOV を考慮した角度 を計算する
    def _calc_angle(self, frame, offset, HFOV):
        return math.atan(math.tan(HFOV / 2.0) * offset / (frame.shape[1] / 2.0))

    # ROI 内の深度情報を取得し、物体の x, y, z 座標を計算する
    def calc_spatials(self, depthData, roi, averaging_method=np.mean):

        depthFrame = depthData.getFrame()

        roi = self._check_input(roi, depthFrame) # If point was passed, convert it to ROI
        xmin, ymin, xmax, ymax = roi

        # Calculate the average depth in the ROI.
        depthROI = depthFrame[ymin:ymax, xmin:xmax]
        inRange = (self.THRESH_LOW <= depthROI) & (depthROI <= self.THRESH_HIGH)
        
        # Required information for calculating spatial coordinates on the host
        HFOV = np.deg2rad(self.calibData.getFov(dai.CameraBoardSocket(depthData.getInstanceNum()), useSpec=False))


        averageDepth = averaging_method(depthROI[inRange])


        # 物体の画面上の位置を取得 ROIの中心座標(centroid)を計算
        centroid = { # Get centroid of the ROI
            'x': int((xmax + xmin) / 2),
            'y': int((ymax + ymin) / 2)
        }

        #  x / y のオフセットを計算 画像の中央を基準にして、物体の x, y オフセットを求める
        midW = int(depthFrame.shape[1] / 2) # middle of the depth img width
        midH = int(depthFrame.shape[0] / 2) # middle of the depth img height
        bb_x_pos = centroid['x'] - midW
        bb_y_pos = centroid['y'] - midH

        angle_x = self._calc_angle(depthFrame, bb_x_pos, HFOV)
        angle_y = self._calc_angle(depthFrame, bb_y_pos, HFOV)

        spatials = {
            'z': averageDepth,
            'x': averageDepth * math.tan(angle_x),
            'y': -averageDepth * math.tan(angle_y)
        }
        return spatials, centroid