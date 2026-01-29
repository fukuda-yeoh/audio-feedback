import synthizer
import time
import numpy as np
import pygame
import csv
import os
import random
from datetime import datetime

# ==============================================================================
# 【シミュレーション基本設定】
# ==============================================================================
SIM_FPS = 30.0          # 疑似フレームレート
DISTANCE_MAX = 10.0     # 音が消える限界距離
BASE_FREQ_DIST = 3.0    # 基準距離

# ウィンドウサイズ（長方形）
SCREEN_WIDTH = 480
SCREEN_HEIGHT = 600

# 【変更点】ボールの移動速度を一定値で固定 (m/s)
CONSTANT_SPEED = 5.0    # ここに好きな速度を入力してください

SCALE = 40              # 1mあたりのピクセル数
# ==============================================================================

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def calculate_pitch_factor(distance):
    # 距離3.0mで1.0倍、0.0mで3.33倍
    return np.interp(distance, [0, 3.0], [3.33, 1.0])

def save_to_csv(data):
    log_dir = "simulation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filename = os.path.join(log_dir, "sim_results.csv")
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["Timestamp", "FPS", "Speed", "X_Pos", "Error_Z(m)", "Lag(s)"])
        writer.writerow(data)

def run_single_trial(generator, source):
    # 設定した一定速度を使用
    speed = CONSTANT_SPEED
    # 横位置(X)のみ、毎回少し変えないと実験にならないためランダムにしています
    obj_x = round(random.uniform(1.0, 3.0), 1)
    
    start_z = 10.0
    end_z = -3.0
    current_z = start_z
    
    pass_time_needed = start_z / speed 
    start_time = time.time()
    clicked = False

    print(f"\n[試行開始] 速度:{speed}m/s (固定), 横位置:{obj_x}m")
    
    screen = pygame.display.get_surface()
    running = True
    
    while running and current_z >= end_z:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.MOUSEBUTTONDOWN and not clicked:
                click_time = time.time() - start_time
                lag = click_time - pass_time_needed
                error_z = current_z
                
                result_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    SIM_FPS, speed, obj_x, round(error_z, 3), round(lag, 3)
                ]
                save_to_csv(result_data)
                print(f"  -> 計測完了: 誤差 {error_z:.3f}m / ラグ {lag:.3f}s")
                clicked = True

        # --- 音響更新 ---
        source.position.value = (obj_x, 0.0, current_z)
        dist = np.sqrt(obj_x**2 + current_z**2)
        p_factor = calculate_pitch_factor(dist)
        
        try:
            generator.pitch_bend.value = p_factor
        except Exception:
            generator.pitch_bend.value = 1.0

        # --- 描画（長方形ウィンドウの中心を基準にする） ---
        screen.fill((30, 30, 30))
        center_x = SCREEN_WIDTH // 2
        center_y = SCREEN_HEIGHT // 2
        
        # 赤い線（真横のライン z=0）を中心(center_y)に引く
        pygame.draw.line(screen, (200, 50, 50), (0, center_y), (SCREEN_WIDTH, center_y), 2)
        # ユーザー（自分）を真ん中に
        pygame.draw.circle(screen, (255, 255, 255), (center_x, center_y), 10)
        
        # ボール描画
        obj_x_px = center_x + int(obj_x * SCALE)
        obj_z_px = center_y - int(current_z * SCALE) 
        pygame.draw.circle(screen, (0, 255, 0), (obj_x_px, obj_z_px), 8)

        pygame.display.flip()
        
        current_z -= speed * (1.0 / SIM_FPS)
        time.sleep(1.0 / SIM_FPS)
    
    return True

def main():
    pygame.init()
    # 長方形サイズでウィンドウ作成
    pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Audio Feedback Constant Speed Sim")

    synthizer.initialize()
    try:
        context = synthizer.Context()
        context.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
        
        # multicamera.pyと同じファイル名を使用
        sound_path = os.path.join(get_project_root(), "sound_files", "1000Hz.wav")
        if not os.path.exists(sound_path):
            print(f"Error: {sound_path} not found.")
            return

        buffer = synthizer.Buffer.from_file(sound_path)
        generator = synthizer.BufferGenerator(context)
        generator.buffer.value = buffer
        generator.looping.value = True
        
        source = synthizer.Source3D(context)
        source.add_generator(generator)
        source.distance_model.value = synthizer.DistanceModel.EXPONENTIAL
        source.rolloff.value = 1.0
        source.distance_ref.value = 0.4
        source.distance_max.value = DISTANCE_MAX
        source.play()

        while True:
            if not run_single_trial(generator, source):
                break
            print("次の試行まで3秒待機...")
            time.sleep(3)

    finally:
        pygame.quit()
        synthizer.shutdown()

if __name__ == "__main__":
    main()