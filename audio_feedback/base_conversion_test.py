import pyrealsense2 as rs

def check_devices():
    ctx = rs.context()
    devices = ctx.query_devices()
    
    print(f"検知されたデバイス数: {len(devices)}")
    
    if len(devices) == 0:
        print("カメラが見つかりません。接続を確認してください。")
        return

    for i, dev in enumerate(devices):
        serial = dev.get_info(rs.camera_info.serial_number)
        name = dev.get_info(rs.camera_info.name)
        print(f"  [{i}] 名前: {name}, シリアルNo: {serial}")
import pyrealsense2 as rs
import numpy as np
import cv2

# --- シリアルナンバー設定 (ここに調べて書き込んでください) ---
# ※カメラが1台しかない場合は、同じ番号を入れるか、片方を空文字 "" にしてください
SERIAL_LEFT  = "845112070212" 
SERIAL_RIGHT = "147122071512"

# --- カメラの配置設定 (こめかみ配置を想定) ---
# 左カメラ: 左に10cm, 左（外側）に30度向いていると仮定
CONFIG_LEFT = {
    "offset": [-0.1, 0.0, 0.0],     # 左へ10cm
    "rotation": [0.0, -15.0, 0.0]   # Y軸 -30度 (左向き)
}

# 右カメラ: 右に10cm, 右（外側）に30度向いていると仮定
CONFIG_RIGHT = {
    "offset": [0.1, 0.0, 0.0],      # 右へ10cm
    "rotation": [0.0, 15.0, 0.0]    # Y軸 +30度 (右向き)
}

# ==========================================
# 変換クラス (変更なし)
# ==========================================
class CameraTransformer:
    def __init__(self, position_offset, rotation_angles_deg):
        self.t = np.array(position_offset)
        rx, ry, rz = np.radians(rotation_angles_deg)
        
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        
        self.R = np.dot(Ry, np.dot(Rx, Rz))
        
        self.T_head_camera = np.eye(4)
        self.T_head_camera[0:3, 0:3] = self.R
        self.T_head_camera[0:3, 3] = self.t

    def transform_to_head_coords(self, point_camera):
        p_cam_h = np.append(np.array(point_camera), 1)
        p_head_h = np.dot(self.T_head_camera, p_cam_h)
        return p_head_h[:3]

# ==========================================
# デバイス管理用クラス
# ==========================================
class RealSenseCam:
    def __init__(self, serial, name, config):
        self.serial = serial
        self.name = name
        self.transformer = CameraTransformer(config["offset"], config["rotation"])
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_device(serial)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.mouse_x, self.mouse_y = 320, 240
        self.active = False

    def start(self):
        try:
            self.profile = self.pipeline.start(self.config)
            self.align = rs.align(rs.stream.color)
            self.intrinsics = self.profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics
            self.active = True
            print(f"[{self.name}] 起動成功")
        except Exception as e:
            print(f"[{self.name}] 起動失敗: {e}")

    def stop(self):
        if self.active:
            self.pipeline.stop()

    def get_frame_data(self):
        if not self.active: return None, None
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth = aligned_frames.get_depth_frame()
        color = aligned_frames.get_color_frame()
        if not depth or not color: return None, None
        return depth, np.asanyarray(color.get_data())

    def update_mouse(self, x, y):
        self.mouse_x, self.mouse_y = x, y

# ==========================================
# メイン実行部
# ==========================================
def run_dual_camera_system():
    # カメラの初期化
    cam_left = RealSenseCam(SERIAL_LEFT, "Left Camera", CONFIG_LEFT)
    cam_right = RealSenseCam(SERIAL_RIGHT, "Right Camera", CONFIG_RIGHT)
    
    cam_left.start()
    cam_right.start()

    # マウスコールバック関数
    def mouse_cb_left(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: cam_left.update_mouse(x, y)
    
    def mouse_cb_right(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN: cam_right.update_mouse(x, y)

    if cam_left.active:
        cv2.namedWindow('Left Camera')
        cv2.setMouseCallback('Left Camera', mouse_cb_left)
    if cam_right.active:
        cv2.namedWindow('Right Camera')
        cv2.setMouseCallback('Right Camera', mouse_cb_right)

    print("--- 検証開始 ---")
    print("左右の画面で『同じ物体』をクリックしてください。")
    print("User座標の値が一致すれば、統合成功です。")
    print("'q' で終了")

    try:
        while True:
            # 左カメラ処理
            if cam_left.active:
                depth_l, img_l = cam_left.get_frame_data()
                if depth_l:
                    dist = depth_l.get_distance(cam_left.mouse_x, cam_left.mouse_y)
                    if dist > 0:
                        raw = rs.rs2_deproject_pixel_to_point(cam_left.intrinsics, [cam_left.mouse_x, cam_left.mouse_y], dist)
                        user = cam_left.transformer.transform_to_head_coords(raw)
                        
                        cv2.circle(img_l, (cam_left.mouse_x, cam_left.mouse_y), 5, (0, 0, 255), -1)
                        cv2.putText(img_l, f"Raw: {raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        # 統合座標を黄色で強調
                        cv2.putText(img_l, f"USER: {user[0]:.2f}, {user[1]:.2f}, {user[2]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    
                    cv2.imshow('Left Camera', img_l)

            # 右カメラ処理
            if cam_right.active:
                depth_r, img_r = cam_right.get_frame_data()
                if depth_r:
                    dist = depth_r.get_distance(cam_right.mouse_x, cam_right.mouse_y)
                    if dist > 0:
                        raw = rs.rs2_deproject_pixel_to_point(cam_right.intrinsics, [cam_right.mouse_x, cam_right.mouse_y], dist)
                        user = cam_right.transformer.transform_to_head_coords(raw)
                        
                        cv2.circle(img_r, (cam_right.mouse_x, cam_right.mouse_y), 5, (0, 0, 255), -1)
                        cv2.putText(img_r, f"Raw: {raw[0]:.2f}, {raw[1]:.2f}, {raw[2]:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                        # 統合座標を黄色で強調
                        cv2.putText(img_r, f"USER: {user[0]:.2f}, {user[1]:.2f}, {user[2]:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
                    
                    cv2.imshow('Right Camera', img_r)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cam_left.stop()
        cam_right.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_dual_camera_system()
if __name__ == "__main__":
    check_devices()