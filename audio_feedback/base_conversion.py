import pyrealsense2 as rs
import numpy as np
import cv2

# ==========================================
# 1. 回転対応版 変換クラス (Updated)
# ==========================================
class CameraTransformer:
    def __init__(self, position_offset, rotation_angles_deg):
        """
        Parameters:
        position_offset (list): [x, y, z] (メートル)
        rotation_angles_deg (list): [pitch(x), yaw(y), roll(z)] (度: degree)
        """
        self.t = np.array(position_offset)
        
        # 角度をラジアンに変換
        rx, ry, rz = np.radians(rotation_angles_deg)
        
        # --- 回転行列の作成 (Rotation Matrix) ---
        # X軸回転 (Pitch)
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        # Y軸回転 (Yaw) - ※重要: こめかみ配置での「内向き/外向き」はここ
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        # Z軸回転 (Roll)
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        # --- 回転行列の結合 ---
        # 適用順序 (YXZ): R = Ry * Rx * Rz
        #   1. Rz: Z軸回転 (Roll  - カメラの傾き)
        #   2. Rx: X軸回転 (Pitch - カメラの上下向き)
        #   3. Ry: Y軸回転 (Yaw   - カメラの左右向き ← 水平設置では主要)
        self.R = Ry @ Rx @ Rz

    def transform_to_head_coords(self, point_camera):
        # 【変換の手順: 回転 → 移動】
        # p_head = R @ p_camera + t
        #
        # Step 1: 回転 - カメラ座標系の向きを頭部座標系の向きに揃える
        p_rotated = self.R @ np.array(point_camera)
        # Step 2: 移動 - カメラ原点（Left IRセンサー）の頭部座標系上の位置を加算
        p_head = p_rotated + self.t
        return p_head

# グローバル変数
mouse_x, mouse_y = 320, 240

def mouse_callback(event, x, y, flags, param):
    global mouse_x, mouse_y
    if event == cv2.EVENT_LBUTTONDOWN:
        mouse_x, mouse_y = x, y

# ==========================================
# 2. メイン実行部
# ==========================================
def run_realsense_rotation_test():
    OFFSET = [0.2, 0.0, 0.0] 
    ROTATION = [0.0, 30.0, 0.0] 
    
    transformer = CameraTransformer(OFFSET, ROTATION)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    align_to = rs.stream.color
    align = rs.align(align_to)
    depth_intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().intrinsics

    print("--- Step 2: 回転補正の検証 ---")
    print(f"現在の設定 -> 位置: {OFFSET}, 回転(度): {ROTATION}")
    print("画面をクリックして座標を確認してください。")

    cv2.namedWindow('Rotation Verification')
    cv2.setMouseCallback('Rotation Verification', mouse_callback)

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            dist = depth_frame.get_distance(mouse_x, mouse_y)

            if dist > 0:
                raw_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mouse_x, mouse_y], dist)
                corrected_point = transformer.transform_to_head_coords(raw_point)

                text_raw = f"Raw:  {raw_point[0]:.2f}, {raw_point[1]:.2f}, {raw_point[2]:.2f}"
                text_usr = f"User: {corrected_point[0]:.2f}, {corrected_point[1]:.2f}, {corrected_point[2]:.2f}"
                
                cv2.rectangle(color_image, (10, 10), (450, 100), (50, 50, 50), -1)
                cv2.putText(color_image, f"Rot: {ROTATION} deg", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(color_image, text_raw, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                cv2.putText(color_image, text_usr, (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(color_image, "Measuring...", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.line(color_image, (mouse_x - 10, mouse_y), (mouse_x + 10, mouse_y), (0, 255, 0), 2)
            cv2.line(color_image, (mouse_x, mouse_y - 10), (mouse_x, mouse_y + 10), (0, 255, 0), 2)
            cv2.imshow('Rotation Verification', color_image)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_realsense_rotation_test()