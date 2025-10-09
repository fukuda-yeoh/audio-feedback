import pyrealsense2 as rs
import numpy as np
import cv2 as cv
import time
from ultralytics import YOLO # YOLOモデルのにultralyticsをインポート

# --- カメラ設定 ---
RESOLUTION_WIDTH = 640
RESOLUTION_HEIGHT = 480
FRAME_RATE = 30

# --- 軸補正のための回転行列と並進ベクトル ---
# 「人を中心とした座標系」の原点 (グローバル座標系内での位置)
PERSON_CENTRIC_ORIGIN_GLOBAL = np.array([0.00, 0.00, 0.39], dtype=np.float32)

# 右のカメラの原点 (グローバル座標系内での位置)
RIGHT_CAMERA_ORIGIN_GLOBAL = np.array([-0.14, 0.00, 0.33], dtype=np.float32)

# 回転行列 R_cam_to_person_centric の計算に使用する固定データポイント
# P_person_obj_target: 人を中心とした座標系での目標物体の位置
P_person_obj_target_for_calib = np.array([0.00, 0.00, 0.30], dtype=np.float32)
# P_cam_obj_measured: 回転後の右カメラのローカル座標系で測定された物体の位置
# ここを前回の出力値に更新します。
P_cam_obj_measured_for_calib = np.array([-0.19, -0.02, 0.22], dtype=np.float32) 

# 並進ベクトル T_cam_to_person_centric
# カメラの原点から「人を中心とした座標系」の原点へのベクトル
# これは、PERSON_CENTRIC_ORIGIN_GLOBAL - RIGHT_CAMERA_ORIGIN_GLOBAL で計算されます。
TRANSLATION_VECTOR_CAM_TO_PERSON_CENTRIC = PERSON_CENTRIC_ORIGIN_GLOBAL - RIGHT_CAMERA_ORIGIN_GLOBAL
# 計算結果: (0.14, 0.00, 0.06)

# 変換後の目標位置 (回転計算用): P_target_after_translation = P_person_obj_target_for_calib - TRANSLATION_VECTOR_CAM_TO_PERSON_CENTRIC
P_target_after_translation_for_rotation = P_person_obj_target_for_calib - TRANSLATION_VECTOR_CAM_TO_PERSON_CENTRIC
# 計算結果: (0.00 - 0.14, 0.00 - 0.00, 0.30 - 0.06) = (-0.14, 0.00, 0.24)


# Y軸周りの回転角を計算
# atan2(Z, X) でX-Z平面上のベクトルの角度を求める
source_vec_xz = np.array([P_cam_obj_measured_for_calib[0], P_cam_obj_measured_for_calib[2]]) # P_cam_obj_measured の XZ 成分
target_vec_xz = np.array([P_target_after_translation_for_rotation[0], P_target_after_translation_for_rotation[2]]) # P_target_after_translation の XZ 成分

angle_source = np.arctan2(source_vec_xz[1], source_vec_xz[0])
angle_target = np.arctan2(target_vec_xz[1], target_vec_xz[0])

# 変換に必要な回転角 (ターゲット - ソース)
rotation_angle_radians = angle_target - angle_source

# Y軸周りの回転行列 (RealSenseのY軸は下向き)
# R_y(theta) = [[cos(theta), 0, sin(theta)], [0, 1, 0], [-sin(theta), 0, cos(theta)]]
ROTATION_MATRIX_CAM_TO_PERSON_CENTRIC = np.array([
    [np.cos(rotation_angle_radians), 0, np.sin(rotation_angle_radians)],
    [0, 1, 0],
    [-np.sin(rotation_angle_radians), 0, np.cos(rotation_angle_radians)]
], dtype=np.float32)

print(f"Calculated Rotation Angle (Y-axis): {np.degrees(rotation_angle_radians):.2f} degrees")
print(f"Calculated Translation Vector: {TRANSLATION_VECTOR_CAM_TO_PERSON_CENTRIC}")
print(f"Calculated Rotation Matrix:\n{ROTATION_MATRIX_CAM_TO_PERSON_CENTRIC}")


# --- YOLOモデルのロード ---
# YOLOモデルのパスを適切に設定してください
# 例: 'yolov8n.pt' (公式の事前学習済みモデル) または 'yolo_model/runs/detect/train4/weights/best.pt' (あなたの学習済みモデル)
try:
    yolo_model = YOLO('yolo_model/runs/detect/train4/weights/best.pt') 
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    print("Please ensure the YOLO model path is correct.")
    exit()

# --- RealSenseパイプラインの設定 ---
def setup_realsense_pipeline(width, height, fps):
    """
    RealSenseパイプラインを設定し、カラーおよび深度ストリームを有効にする。
    最初の検出されたRealSenseカメラを使用します。
    """
    pipeline = rs.pipeline()
    config = rs.config()

    # 接続されているデバイスを検索し、最初のデバイスを使用
    ctx = rs.context()
    if len(ctx.devices) == 0:
        print("No Intel RealSense devices connected. Exiting.")
        return None, None
    
    # 最初のデバイスのシリアル番号を取得
    first_serial = ctx.devices[0].get_info(rs.camera_info.serial_number)
    print(f"Using first detected device: {first_serial}")
    config.enable_device(first_serial)

    # カラーと深度ストリームを有効化
    config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
    config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

    # パイプラインを開始
    profile = pipeline.start(config)
    print("RealSense pipeline started.")

    # カメラが安定するまで少し待機
    time.sleep(1.5) 
    return pipeline, profile

# --- 3D位置の計算と表示 ---
def process_frames_and_detect(pipeline, yolo_model, rotation_matrix, translation_vector):
    """
    RealSenseカメラからフレームを取得し、YOLOで物体を検出し、
    検出された物体の3D位置を計算し、指定された回転と並進で補正して表示・出力する。
    rotation_matrix: カメラのローカル座標系から目的の座標系への回転行列
    translation_vector: カメラのローローカル座標系から目的の座標系への並進ベクトル
    """
    # 深度とカラーのアラインメント設定
    align_to = rs.stream.color
    align = rs.align(align_to)

    # PointCloudオブジェクトを初期化
    pc = rs.pointcloud()

    print("\n--- Starting Object Detection and 3D Position Estimation (Corrected Frame) ---")
    print("Press 'q' or 'Esc' to quit.")

    try:
        while True:
            # フレームセットを取得
            frames = pipeline.wait_for_frames(5000) # タイムアウトは必要に応じて調整

            # 深度フレームをカラーフレームにアラインメント
            aligned_frames = align.process(frames)
            color_frame = aligned_frames.get_color_frame()
            depth_frame = aligned_frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            # NumPy配列に変換
            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) # 深度画像（ミリメートル単位）

            # YOLO推論 (カラー画像に対して行う)
            results = yolo_model(color_image, verbose=False)

            # 結果描画 (検出結果が描画された画像)
            annotated_frame = results[0].plot()

            # PointCloudを生成し、3D座標データを取得
            pc.map_to(color_frame) # PointCloudをカラーフレームにマップ
            points = pc.calculate(depth_frame) # 深度フレームからポイントクラウドを計算
            # get_vertices() の結果を np.void 型として受け取り、その後 float32 としてビュー
            vtx = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3) # (N, 3) の3D座標配列 (メートル単位)

            detected_objects_info = []

            # 検出されたオブジェクトから3D位置情報を取得
            for r in results: # 各検出結果について
                for box in r.boxes: # 各バウンディングボックスについて
                    class_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = yolo_model.names[class_id] # クラス名を取得

                    # バウンディングボックスの座標を取得 (xyxy形式: [x1, y1, x2, y2])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # バウンディングボックスの中心座標を計算 (カラー画像のピクセル座標)
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)

                    # 中心座標が画像の範囲内にあるか確認
                    if 0 <= center_x < RESOLUTION_WIDTH and 0 <= center_y < RESOLUTION_HEIGHT:
                        # 3D座標のインデックスを計算
                        # vtx配列は画像ピクセルをフラット化したものに対応
                        pixel_index = center_y * RESOLUTION_WIDTH + center_x

                        if 0 <= pixel_index < len(vtx):
                            # カメラローカル座標系での3D位置を取得 (メートル単位)
                            # RealSenseの座標系: X右、Y下、Z手前 (カメラから離れる方向)
                            point_3d_camera_coord = vtx[pixel_index] 
                            
                            # 深度値が有効な数値であり、かつ0でないことを確認
                            if not np.isfinite(point_3d_camera_coord[2]) or point_3d_camera_coord[2] == 0.0:
                                continue

                            # ここで回転と並進補正を適用
                            # P_person = R * P_cam + T
                            point_3d_corrected_coord = np.dot(rotation_matrix, point_3d_camera_coord) + translation_vector

                            detected_objects_info.append(
                                {
                                    "class": class_name,
                                    "confidence": conf,
                                    "bbox": (x1, y1, x2, y2),
                                    "3d_position_m": point_3d_corrected_coord.tolist() # [X, Y, Z] メートル単位
                                }
                            )

                            # 描画されたフレームに3D位置情報を追加 (補正後の値)
                            text_x = x1
                            text_y = y1 - 10 if y1 - 10 > 10 else y1 + 20 # バウンディングボックスの上か下

                            cv.putText(annotated_frame,
                                       f"X:{point_3d_corrected_coord[0]:.2f} Y:{point_3d_corrected_coord[1]:.2f} Z:{point_3d_corrected_coord[2]:.2f}m",
                                       (text_x, text_y),
                                       cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            # 検出されたオブジェクトの3D位置情報をコンソールに出力 (補正後の値)
            if detected_objects_info:
                print(f"\n--- Detected Objects in Corrected Reference Coordinate System ---")
                for obj_info in detected_objects_info:
                    pos = obj_info['3d_position_m']
                    print(f"  Class: {obj_info['class']}, Conf: {obj_info['confidence']:.2f}, Pos: (X:{pos[0]:.2f}m, Y:{pos[1]:.2f}m, Z:{pos[2]:.2f}m)")
            else:
                print("\nNo objects detected in this frame.")


            # ウィンドウに表示
            cv.imshow("RealSense YOLO 3D Position (Corrected Frame)", annotated_frame)

            key = cv.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' または 'Esc' で終了
                print("Exiting program...")
                break

    finally:
        cv.destroyAllWindows()


# --- メインプログラム ---
def main():
    pipeline, profile = setup_realsense_pipeline(RESOLUTION_WIDTH, RESOLUTION_HEIGHT, FRAME_RATE)
    
    if pipeline is None: # カメラが見つからない場合は終了
        return

    try:
        # process_frames_and_detect関数に計算した回転行列と並進ベクトルを渡す
        process_frames_and_detect(pipeline, yolo_model, 
                                  ROTATION_MATRIX_CAM_TO_PERSON_CENTRIC, 
                                  TRANSLATION_VECTOR_CAM_TO_PERSON_CENTRIC)
    finally:
        # パイプラインを停止
        pipeline.stop()
        print("RealSense pipeline stopped.")

if __name__ == "__main__":
    main()
