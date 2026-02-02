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

# --- 画面設定 ---
SINGLE_SCREEN_WIDTH = 480
SCREEN_HEIGHT = 600
TOTAL_WIDTH = SINGLE_SCREEN_WIDTH * 2

USER_Y_POS = int(SCREEN_HEIGHT * 0.8)

# --- ボール移動設定 ---
CONSTANT_SPEED = 5.0    # 移動速度 (m/s)
START_Z = 10.0          # 開始位置
END_Z = -2.0            # 終了位置

# --- 音響設定 ---
DISTANCE_MAX = 15.0     
AUDIO_ROLLOFF = 0.5     
BASE_FREQ_DIST = 3.0    
PITCH_MAX_MULT = 2.0    

SCALE = 40              
# ==============================================================================

# 実験開始時にファイル名を決定（グローバル変数として保持）
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"experiment_{current_time_str}.csv"

def get_project_root():
    return os.path.dirname(os.path.abspath(__file__))

def calculate_pitch_factor(distance):
    if distance > BASE_FREQ_DIST:
        return 1.0
    return np.interp(distance, [0, BASE_FREQ_DIST], [PITCH_MAX_MULT, 1.0])

def create_beep_buffer(ctx, freq, duration=0.1):
    sr = 44100
    samples = int(sr * duration)
    t = np.arange(samples) / sr
    wave = 0.5 * np.sin(2 * np.pi * freq * t)
    fade_len = int(sr * 0.01)
    wave[:fade_len] *= np.linspace(0, 1, fade_len)
    wave[-fade_len:] *= np.linspace(1, 0, fade_len)
    return synthizer.Buffer.from_float_array(sr, 1, wave.astype(np.float32))

def save_to_csv(data):
    """データをCSVに保存（毎回新規ファイルまたは追記）"""
    log_dir = "simulation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # グローバルで決めたファイル名を使用
    filepath = os.path.join(log_dir, CSV_FILENAME)
    file_exists = os.path.isfile(filepath)
    
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # ファイルが新規作成された時だけヘッダーを書き込む
        if not file_exists:
            writer.writerow([
                "Timestamp",      # 日時
                "FPS",            # フレームレート
                "Speed(m/s)",     # 速度
                "True_X(m)",      # 正解の横位置
                "Clicked_X(m)",   # 回答した横位置
                "Error_X(m)",     # 横方向の誤差
                "Error_Z(m)",     # 縦方向(タイミング)の誤差
                "Lag(s)"          # 反応時間ラグ
            ])
            
        writer.writerow(data)

def draw_dual_screen(screen, font, state_text, obj_x, current_z, show_ball_on_admin=True):
    screen.fill((0, 0, 0))
    
    local_center_x = SINGLE_SCREEN_WIDTH // 2
    user_screen_y = USER_Y_POS

    # --- 左側：管理者用 ---
    pygame.draw.rect(screen, (30, 30, 30), (0, 0, SINGLE_SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.draw.line(screen, (200, 50, 50), (0, user_screen_y), (SINGLE_SCREEN_WIDTH, user_screen_y), 2)
    pygame.draw.line(screen, (100, 100, 100), (SINGLE_SCREEN_WIDTH, 0), (SINGLE_SCREEN_WIDTH, SCREEN_HEIGHT), 2)
    pygame.draw.circle(screen, (255, 255, 255), (local_center_x, user_screen_y), 10)
    
    if show_ball_on_admin:
        ball_screen_y = user_screen_y - int(current_z * SCALE)
        ball_screen_x = local_center_x + int(obj_x * SCALE)
        pygame.draw.circle(screen, (0, 255, 0), (ball_screen_x, ball_screen_y), 8)
    
    info_surf = font.render(f"Admin: X={obj_x:.1f}m Z={current_z:.1f}m", True, (150, 150, 150))
    screen.blit(info_surf, (10, 10))

    # --- 右側：被験者用 ---
    offset = SINGLE_SCREEN_WIDTH
    pygame.draw.line(screen, (100, 50, 50), (offset, user_screen_y), (TOTAL_WIDTH, user_screen_y), 1)
    pygame.draw.circle(screen, (255, 255, 255), (offset + local_center_x, user_screen_y), 10)
    
    if state_text:
        text_surf = font.render(state_text, True, (255, 255, 0))
        text_rect = text_surf.get_rect(center=(offset + local_center_x, user_screen_y - 100))
        screen.blit(text_surf, text_rect)
        
        admin_text_rect = text_surf.get_rect(center=(local_center_x, user_screen_y - 100))
        screen.blit(text_surf, admin_text_rect)

    pygame.display.flip()

def run_countdown(screen, font, ctx, beep_low, beep_high, obj_x, start_z):
    ui_source = synthizer.DirectSource(ctx)
    counts = ["3", "2", "1", "START!"]
    
    for text in counts:
        draw_dual_screen(screen, font, text, obj_x, start_z, show_ball_on_admin=True)
        
        gen = synthizer.BufferGenerator(ctx)
        gen.buffer.value = beep_high if text == "START!" else beep_low
        ui_source.add_generator(gen)
        
        time.sleep(1.0)
        ui_source.remove_generator(gen)

def run_single_trial(ctx, generator, source, beep_low, beep_high, font):
    speed = CONSTANT_SPEED
    
    # 左右ランダム (-1 or 1) * 距離
    side = random.choice([-1, 1])
    dist = random.uniform(1.0, 3.0)
    obj_x = round(side * dist, 2)
    
    current_z = START_Z
    pass_time_needed = START_Z / speed 
    
    screen = pygame.display.get_surface()
    local_center_x = SINGLE_SCREEN_WIDTH // 2
    
    # 1. カウントダウン
    source.pause()
    run_countdown(screen, font, ctx, beep_low, beep_high, obj_x, START_Z)
    
    # 2. 計測開始
    print(f"\n[試行開始] 速度:{speed}m/s, 位置X:{obj_x}m")
    start_time = time.time()
    clicked = False
    running = True
    
    source.play()

    while running and current_z >= END_Z:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            
            if event.type == pygame.MOUSEBUTTONDOWN and not clicked:
                mouse_x, mouse_y = event.pos
                
                # 被験者画面(右)のクリック位置計算
                rel_x_px = mouse_x - SINGLE_SCREEN_WIDTH
                if rel_x_px < 0: rel_x_px = mouse_x
                
                diff_px = rel_x_px - local_center_x
                clicked_x_meter = diff_px / SCALE
                
                click_time = time.time() - start_time
                lag = click_time - pass_time_needed
                error_z = current_z
                error_x = clicked_x_meter - obj_x
                
                # --- ここでデータを保存 ---
                result_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    SIM_FPS,         # ここでFPSを保存
                    speed,           # 速度
                    obj_x,           # 正解X
                    round(clicked_x_meter, 3), # 回答X
                    round(error_x, 3),         # 誤差X
                    round(error_z, 3),         # 誤差Z
                    round(lag, 3)              # ラグ
                ]
                save_to_csv(result_data)
                
                print(f"  -> 計測: 正解X={obj_x}m, 回答X={clicked_x_meter:.2f}m")
                clicked = True
                source.pause()

        if not clicked:
            source.position.value = (obj_x, 0.0, current_z)
            dist = np.sqrt(obj_x**2 + current_z**2)
            p_factor = calculate_pitch_factor(dist)
            try:
                generator.pitch_bend.value = p_factor
            except Exception:
                generator.pitch_bend.value = 1.0

        status_text = "CLICKED!" if clicked else ""
        draw_dual_screen(screen, font, status_text, obj_x, current_z, show_ball_on_admin=True)
        
        current_z -= speed * (1.0 / SIM_FPS)
        time.sleep(1.0 / SIM_FPS)
    
    source.pause()
    return True

def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("arial", 36, bold=True)
    
    screen = pygame.display.set_mode((TOTAL_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"Experiment: {CSV_FILENAME}")

    synthizer.initialize()
    try:
        ctx = synthizer.Context()
        ctx.default_panner_strategy.value = synthizer.PannerStrategy.HRTF
        
        beep_low = create_beep_buffer(ctx, 440.0)
        beep_high = create_beep_buffer(ctx, 880.0)
        
        sound_path = os.path.join(get_project_root(), "sound_files", "1000Hz.wav")
        if not os.path.exists(sound_path):
            print(f"Error: {sound_path} not found.")
            return

        buffer = synthizer.Buffer.from_file(sound_path)
        generator = synthizer.BufferGenerator(ctx)
        generator.buffer.value = buffer
        generator.looping.value = True
        
        source = synthizer.Source3D(ctx)
        source.add_generator(generator)
        source.distance_model.value = synthizer.DistanceModel.INVERSE
        source.rolloff.value = AUDIO_ROLLOFF
        source.distance_ref.value = 1.0 
        source.distance_max.value = DISTANCE_MAX
        source.pause()

        running_main = True
        while running_main:
            # 待機画面に現在のFPSを表示
            msg = f"SPACE to Start (FPS:{SIM_FPS})"
            draw_dual_screen(screen, font, msg, 0, START_Z, show_ball_on_admin=False)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_main = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        if not run_single_trial(ctx, generator, source, beep_low, beep_high, font):
                            running_main = False
                        time.sleep(0.5)
                    if event.key == pygame.K_ESCAPE:
                        running_main = False
            
            time.sleep(0.1)

    finally:
        pygame.quit()
        synthizer.shutdown()
        print(f"\n実験終了。データは {CSV_FILENAME} に保存されました。")

if __name__ == "__main__":
    main()