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
        
        # 回転順序: R = Ry * Rx * Rz (Y軸回転を主軸とみなす一般的な順序)
        # ※必要に応じて順序は調整可能ですが、通常はこれで十分です
        self.R = np.dot(Ry, np.dot(Rx, Rz))
        
        # --- 同次変換行列 T の結合 ---
        # T = [ R  t ]
        #     [ 0  1 ]
        self.T_head_camera = np.eye(4)
        self.T_head_camera[0:3, 0:3] = self.R  # 回転成分
        self.T_head_camera[0:3, 3] = self.t    # 平行移動成分

    def transform_to_head_coords(self, point_camera):
        p_cam_h = np.append(np.array(point_camera), 1)
        p_head_h = np.dot(self.T_head_camera, p_cam_h)
        return p_head_h[:3]

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
    # --- 【設定】 Step 2 のための設定 ---
    
    # 位置: 右に20cm (Step 1と同じ)
    OFFSET = [0.2, 0.0, 0.0] 
    
    # 角度: [Pitch(X), Yaw(Y), Roll(Z)]
    # ★ここを変えて検証します★
    # 例: Y軸(縦軸)を中心に 45度 回転させた場合
    ROTATION = [0.0, 30.0, 0.0] 
    
    transformer = CameraTransformer(OFFSET, ROTATION)

    # --- RealSense初期化 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    profile = pipeline.start(config)
    
    # 深度調整用
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
                # 1. 生座標
                raw_point = rs.rs2_deproject_pixel_to_point(depth_intrinsics, [mouse_x, mouse_y], dist)
                
                # 2. 補正後座標 (回転 + 平行移動)
                corrected_point = transformer.transform_to_head_coords(raw_point)

                # 表示 (小数点2桁)
                text_raw = f"Raw:  {raw_point[0]:.2f}, {raw_point[1]:.2f}, {raw_point[2]:.2f}"
                text_usr = f"User: {corrected_point[0]:.2f}, {corrected_point[1]:.2f}, {corrected_point[2]:.2f}"
                
                # 情報表示エリアを描画
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