"""
rotation_solver.py

Step2（回転なし）とStep3（回転あり）のCSVから、カメラの回転角度を逆算する。

【原理】
同一物体を同じ位置から回転なし(Step2)と回転あり(Step3)で計測すると:
  p_step2 = R(θ) @ p_step3

Y軸回転(Yaw)の場合、θ の解析解:
  θ_y = atan2(z3*x2 - x3*z2,  x3*x2 + z3*z2)

X軸回転(Pitch)の場合:
  θ_x = atan2(y3*z2 - z3*y2,  y3*y2 + z3*z2)

使い方:
  python rotation_solver.py <step2_csv> <step3_csv>   # ファイル指定
  python rotation_solver.py                            # logsフォルダから自動検出
"""

import numpy as np
import csv
import sys
import os
import glob
import json


CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "validation_config.json")


def read_raw_coords(csv_path):
    points = []
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                x, y, z = float(row["Raw_X"]), float(row["Raw_Y"]), float(row["Raw_Z"])
                if z > 0 and not np.isnan(z):
                    points.append([x, y, z])
            except (ValueError, KeyError):
                continue
    return np.array(points) if points else None


def median_point(points):
    return np.nanmedian(points, axis=0)


def calc_yaw(p2, p3):
    """
    p2 = Ry(θ) @ p3 を満たす Yaw角 θ を返す（度）
    使用軸: XZ平面
    """
    x2, _, z2 = p2
    x3, _, z3 = p3
    return np.degrees(np.arctan2(z3 * x2 - x3 * z2, x3 * x2 + z3 * z2))


def calc_pitch(p2, p3):
    """
    p2 = Rx(θ) @ p3 を満たす Pitch角 θ を返す（度）
    使用軸: YZ平面
    """
    _, y2, z2 = p2
    _, y3, z3 = p3
    return np.degrees(np.arctan2(y3 * z2 - z3 * y2, y3 * y2 + z3 * z2))


def find_step_csv(step_num):
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    pattern = os.path.join(log_dir, f"validation_step{step_num}_*.csv")
    files = sorted(glob.glob(pattern))
    return files[-1] if files else None


def main():
    if len(sys.argv) == 3:
        step2_csv, step3_csv = sys.argv[1], sys.argv[2]
    else:
        step2_csv = find_step_csv(2)
        step3_csv = find_step_csv(3)
        if not step2_csv or not step3_csv:
            print("エラー: logsフォルダにvalidation_step2_*.csv と validation_step3_*.csv が必要です。")
            print("使い方: python rotation_solver.py <step2.csv> <step3.csv>")
            return
        print(f"自動検出:")
        print(f"  Step2: {os.path.basename(step2_csv)}")
        print(f"  Step3: {os.path.basename(step3_csv)}")

    pts2 = read_raw_coords(step2_csv)
    pts3 = read_raw_coords(step3_csv)

    if pts2 is None or len(pts2) == 0:
        print(f"エラー: {step2_csv} から座標を読み込めません。")
        return
    if pts3 is None or len(pts3) == 0:
        print(f"エラー: {step3_csv} から座標を読み込めません。")
        return

    p2 = median_point(pts2)
    p3 = median_point(pts3)

    print(f"\n{'='*55}")
    print(f"Step 2（回転なし） - 中央値 Raw座標  ({len(pts2)} サンプル)")
    print(f"  X={p2[0]:+.4f}m  Y={p2[1]:+.4f}m  Z={p2[2]:+.4f}m")
    print(f"\nStep 3（回転あり） - 中央値 Raw座標  ({len(pts3)} サンプル)")
    print(f"  X={p3[0]:+.4f}m  Y={p3[1]:+.4f}m  Z={p3[2]:+.4f}m")
    print(f"{'='*55}")

    yaw   = calc_yaw(p2, p3)
    pitch = calc_pitch(p2, p3)

    print(f"\n【逆算された回転角度】")
    print(f"  Yaw  (Y軸回転): {yaw:+.2f}°   ← カメラの左右向き")
    print(f"  Pitch (X軸回転): {pitch:+.2f}°   ← カメラの上下向き")
    print()
    print(f"  符号の見方: 正 = 右向き（時計回り）, 負 = 左向き（反時計回り）")

    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        configured_rotation = cfg.get("rotation", [0, 0, 0])
        print(f"\n【設定値との比較】")
        print(f"  validation_config.json rotation: {configured_rotation}")
        print(f"  逆算 Yaw:   {yaw:+.2f}° vs 設定 Yaw: {configured_rotation[1]:+.2f}°  "
              f"→ 差: {yaw - configured_rotation[1]:+.2f}°")
        print(f"  逆算 Pitch: {pitch:+.2f}° vs 設定 Pitch: {configured_rotation[0]:+.2f}°  "
              f"→ 差: {pitch - configured_rotation[0]:+.2f}°")

    print(f"{'='*55}")


if __name__ == "__main__":
    main()