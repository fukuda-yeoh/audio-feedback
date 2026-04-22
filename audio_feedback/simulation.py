import synthizer
import time
import numpy as np
import pygame
import csv
import os
import json
import random
import math
from datetime import datetime

# ==============================================================================
# 【シミュレーション基本設定】
# ==============================================================================
SIM_FPS = 30.0

SINGLE_SCREEN_WIDTH = 480
SCREEN_HEIGHT = 600
TOTAL_WIDTH = SINGLE_SCREEN_WIDTH * 2

USER_Y_POS = int(SCREEN_HEIGHT * 0.8)

CONSTANT_SPEED = 5.0
START_Z = 10.0
END_Z = -2.0

DISTANCE_MAX = 15.0
AUDIO_ROLLOFF = 0.5
BASE_FREQ_DIST = 3.0
PITCH_MAX_MULT = 2.0

SCALE = 40

# ==============================================================================
# 【カラー定数 - アーケード風テーマ】
# ==============================================================================
BG_TOP      = (255,  80, 120)   # ピンク
BG_BOT      = (255, 160,   0)   # オレンジ
STAR_COL    = (255, 235,  80)   # 黄色い星
LINE_COL    = (255, 255, 255)   # 白仕切り線
CAT_BODY    = (255, 240, 200)   # クリーム色
CAT_EAR     = (255, 180, 200)   # ピンク内耳
CAT_EYE     = ( 40,  40,  40)   # 目
TITLE_COL   = (255, 255, 255)   # タイトル文字
TITLE_SHD   = (180,  30,  80)   # タイトル影
SCORE_BG    = ( 40,   0,  70)   # スコア画面背景（深紫）
SCORE_COL   = (255, 215,   0)   # スコアタイトル（金）
SCORE_SHD   = (160,  80,   0)   # スコアタイトル影

# 星の位置（main()で初期化）
_STARS = []

# ==============================================================================

current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"experiment_{current_time_str}.csv"
SCORES_FILE = os.path.join("simulation_logs", "best_scores.json")


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
    log_dir = "simulation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    filepath = os.path.join(log_dir, CSV_FILENAME)
    file_exists = os.path.isfile(filepath)
    with open(filepath, mode='a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow([
                "Timestamp", "FPS", "Speed(m/s)",
                "True_X(m)", "Clicked_X(m)",
                "Error_X(m)", "Error_Z(m)", "Lag(s)"
            ])
        writer.writerow(data)


# ==============================================================================
# 【スコア管理】
# ==============================================================================
def load_scores():
    if not os.path.exists(SCORES_FILE):
        return []
    try:
        with open(SCORES_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_score(error_x, error_z):
    log_dir = "simulation_logs"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    scores = load_scores()
    total = math.sqrt(error_x ** 2 + error_z ** 2)
    entry = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "error_x": round(error_x, 3),
        "error_z": round(error_z, 3),
        "total_error": round(total, 3),
    }
    scores.append(entry)
    scores.sort(key=lambda s: s["total_error"])
    scores = scores[:5]
    with open(SCORES_FILE, 'w', encoding='utf-8') as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


# ==============================================================================
# 【描画ユーティリティ】
# ==============================================================================
def _star_polygon(cx, cy, r_outer, r_inner, n=5):
    pts = []
    for i in range(n * 2):
        angle = math.pi / n * i - math.pi / 2
        r = r_outer if i % 2 == 0 else r_inner
        pts.append((int(cx + r * math.cos(angle)), int(cy + r * math.sin(angle))))
    return pts


def init_stars():
    global _STARS
    random.seed(42)
    _STARS = [
        (random.randint(20, TOTAL_WIDTH - 20),
         random.randint(20, SCREEN_HEIGHT - 20),
         random.randint(6, 12))
        for _ in range(10)
    ]


def draw_gradient_bg(surface, x, y, w, h):
    for dy in range(h):
        t = dy / max(h - 1, 1)
        r = int(BG_TOP[0] + (BG_BOT[0] - BG_TOP[0]) * t)
        g = int(BG_TOP[1] + (BG_BOT[1] - BG_TOP[1]) * t)
        b = int(BG_TOP[2] + (BG_BOT[2] - BG_TOP[2]) * t)
        pygame.draw.line(surface, (r, g, b), (x, y + dy), (x + w - 1, y + dy))


def draw_stars(surface):
    for sx, sy, sz in _STARS:
        pts = _star_polygon(sx, sy, sz, int(sz * 0.4))
        pygame.draw.polygon(surface, STAR_COL, pts)


def draw_cat(surface, cx, cy, message=None, black=False):
    """ポケモン風ネコキャラクターを描画。(cx, cy) は足元中心。black=True で黒猫。"""
    if black:
        BODY    = (25,  20,  35)   # 黒紫
        OUTLINE = (60,  50,  80)
        EAR_IN  = (100,  0,  60)   # 濃いピンク
        EYE_COL = (220, 180,  20)  # 琥珀色の目
        WHISKER = (200, 200, 220)
    else:
        BODY    = CAT_BODY
        OUTLINE = (160, 130, 90)
        EAR_IN  = CAT_EAR
        EYE_COL = (60, 160, 255)
        WHISKER = (255, 255, 255)

    # しっぽ（太め・アウトライン付き）
    tail_pts = [
        (cx + 16, cy - 10),
        (cx + 32, cy - 14),
        (cx + 44, cy - 30),
        (cx + 42, cy - 54),
    ]
    pygame.draw.lines(surface, OUTLINE, False, tail_pts, 7)
    pygame.draw.lines(surface, BODY, False, tail_pts, 4)

    # 胴体（アウトライン→塗り）
    body_out = pygame.Rect(cx - 23, cy - 37, 46, 32)
    body_in  = pygame.Rect(cx - 21, cy - 35, 42, 28)
    pygame.draw.ellipse(surface, OUTLINE, body_out)
    pygame.draw.ellipse(surface, BODY, body_in)

    # 頭（大きめ・ポケモン風）
    pygame.draw.circle(surface, OUTLINE, (cx, cy - 62), 28)
    pygame.draw.circle(surface, BODY, (cx, cy - 62), 26)

    # 耳（外・アウトライン）
    l_ear    = [(cx - 24, cy - 72), (cx - 10, cy - 94), (cx - 2, cy - 70)]
    r_ear    = [(cx + 24, cy - 72), (cx + 10, cy - 94), (cx + 2, cy - 70)]
    l_out    = [(cx - 26, cy - 71), (cx - 11, cy - 97), (cx - 1, cy - 69)]
    r_out    = [(cx + 26, cy - 71), (cx + 11, cy - 97), (cx + 1, cy - 69)]
    pygame.draw.polygon(surface, OUTLINE, l_out)
    pygame.draw.polygon(surface, OUTLINE, r_out)
    pygame.draw.polygon(surface, BODY, l_ear)
    pygame.draw.polygon(surface, BODY, r_ear)
    # 耳（内・ピンク）
    l_inner  = [(cx - 20, cy - 73), (cx - 11, cy - 87), (cx - 5, cy - 71)]
    r_inner  = [(cx + 20, cy - 73), (cx + 11, cy - 87), (cx + 5, cy - 71)]
    pygame.draw.polygon(surface, EAR_IN, l_inner)
    pygame.draw.polygon(surface, EAR_IN, r_inner)

    # 目（ポケモン風：白目＋虹彩＋瞳孔＋ハイライト）
    for ex in [cx - 11, cx + 11]:
        ey = cy - 65
        pygame.draw.circle(surface, OUTLINE,         (ex, ey), 9)
        pygame.draw.circle(surface, (255, 255, 255), (ex, ey), 8)       # 白目
        pygame.draw.circle(surface, EYE_COL,         (ex, ey), 5)       # 虹彩
        pygame.draw.circle(surface, (10, 10, 10),    (ex, ey), 3)       # 瞳孔
        pygame.draw.circle(surface, (255, 255, 255), (ex - 2, ey - 2), 2)  # ハイライト

    # ほっぺ（赤みの楕円・半透明、黒猫は非表示）
    if not black:
        blush = pygame.Surface((16, 10), pygame.SRCALPHA)
        pygame.draw.ellipse(blush, (255, 130, 130, 130), (0, 0, 16, 10))
        surface.blit(blush, (cx - 26, cy - 59))
        surface.blit(blush, (cx + 10, cy - 59))

    # 鼻（ハート風小三角）
    pygame.draw.polygon(surface, (255, 80, 130),
                        [(cx, cy - 57), (cx - 3, cy - 53), (cx + 3, cy - 53)])

    # 口（W型）
    pygame.draw.line(surface, OUTLINE, (cx - 2, cy - 53), (cx - 7, cy - 49), 2)
    pygame.draw.line(surface, OUTLINE, (cx + 2, cy - 53), (cx + 7, cy - 49), 2)

    # ヒゲ（左右各3本）
    for i in range(3):
        wy = cy - 55 + i * 3
        pygame.draw.line(surface, WHISKER, (cx - 5, wy), (cx - 26 + i * 2, wy - i), 1)
        pygame.draw.line(surface, WHISKER, (cx + 5, wy), (cx + 26 - i * 2, wy - i), 1)

    # 吹き出し
    if message:
        _draw_speech_bubble(surface, cx, cy - 92, message)


def _draw_speech_bubble(surface, tip_cx, tip_cy, message):
    """吹き出しを描画。tip_cx/tip_cy はしっぽの先（キャラ頭上）の座標。"""
    bubble_font = pygame.font.SysFont("meiryo", 18, bold=True)
    text_surf = bubble_font.render(message, True, (40, 20, 60))
    tw, th = text_surf.get_size()
    pad = 10
    bw, bh = tw + pad * 2, th + pad * 2

    # バブル左上位置（吹き出しの下辺がtip_cyに来るよう上へオフセット）
    tail_h = 12
    bx = tip_cx - bw // 2
    by = tip_cy - bh - tail_h
    # 画面端クランプ
    bx = max(4, min(bx, TOTAL_WIDTH - bw - 4))

    # 背景（白系丸角矩形）
    bubble_rect = pygame.Rect(bx, by, bw, bh)
    pygame.draw.rect(surface, (255, 252, 220), bubble_rect, border_radius=10)
    pygame.draw.rect(surface, (40, 20, 60),  bubble_rect, 2, border_radius=10)

    # しっぽ三角
    tail_cx = min(max(tip_cx, bx + 12), bx + bw - 12)
    tail_pts = [
        (tail_cx - 7, by + bh - 1),
        (tail_cx + 7, by + bh - 1),
        (tail_cx,     by + bh + tail_h),
    ]
    pygame.draw.polygon(surface, (255, 252, 220), tail_pts)
    pygame.draw.line(surface, (40, 20, 60), tail_pts[0], tail_pts[2], 2)
    pygame.draw.line(surface, (40, 20, 60), tail_pts[2], tail_pts[1], 2)

    surface.blit(text_surf, (bx + pad, by + pad))


# ==============================================================================
# 【メイン描画】
# ==============================================================================
def draw_dual_screen(screen, font, state_text, obj_x, current_z, show_ball_on_admin=True):
    local_center_x = SINGLE_SCREEN_WIDTH // 2
    user_screen_y = USER_Y_POS

    # ── グラデーション背景 ──
    draw_gradient_bg(screen, 0, 0, TOTAL_WIDTH, SCREEN_HEIGHT)
    draw_stars(screen)

    # ── 区切り線 ──
    pygame.draw.line(screen, LINE_COL, (SINGLE_SCREEN_WIDTH, 0), (SINGLE_SCREEN_WIDTH, SCREEN_HEIGHT), 2)

    # ── タイトル（影付き）──
    title_font = pygame.font.SysFont("arial", 26, bold=True)
    for scr_cx in [local_center_x, SINGLE_SCREEN_WIDTH + local_center_x]:
        for ox, oy in [(-2, 2), (2, 2)]:
            sh = title_font.render("AUDIO CATCHER", True, TITLE_SHD)
            screen.blit(sh, sh.get_rect(center=(scr_cx + ox, 22 + oy)))
        ts = title_font.render("AUDIO CATCHER", True, TITLE_COL)
        screen.blit(ts, ts.get_rect(center=(scr_cx, 22)))

    # ── ベースライン ──
    pygame.draw.line(screen, (255, 255, 100), (0, user_screen_y),
                     (SINGLE_SCREEN_WIDTH, user_screen_y), 2)
    pygame.draw.line(screen, (255, 255, 100), (SINGLE_SCREEN_WIDTH, user_screen_y),
                     (TOTAL_WIDTH, user_screen_y), 2)

    # ── ネコキャラクター（状態に応じたセリフ付き）──
    _cat_lines = {
        "3": "さん！", "2": "に！", "1": "いち！",
        "START!": "ゴー！！", "CLICKED!": "ナイス！",
    }
    if state_text in _cat_lines:
        cat_msg = _cat_lines[state_text]
    elif "SPACE" in (state_text or ""):
        cat_msg = "ニャ！はじめよう！"
    else:
        cat_msg = "ニャ～♪"
    draw_cat(screen, local_center_x, user_screen_y, message=cat_msg)
    draw_cat(screen, SINGLE_SCREEN_WIDTH + local_center_x, user_screen_y, message=cat_msg)

    # ── 管理者側：ボール ──
    if show_ball_on_admin:
        ball_screen_y = user_screen_y - int(current_z * SCALE)
        ball_screen_x = local_center_x + int(obj_x * SCALE)
        pygame.draw.circle(screen, (80, 255, 80), (ball_screen_x, ball_screen_y), 8)
        pygame.draw.circle(screen, (255, 255, 255), (ball_screen_x, ball_screen_y), 8, 1)

    # ── 管理者情報 ──
    small_font = pygame.font.SysFont("arial", 17)
    info_surf = small_font.render(f"Admin: X={obj_x:.1f}m  Z={current_z:.1f}m", True, (255, 255, 255))
    screen.blit(info_surf, (8, SCREEN_HEIGHT - 22))

    # ── 状態テキスト ──
    if state_text:
        text_surf = font.render(state_text, True, (255, 255, 0))
        for scr_cx in [local_center_x, SINGLE_SCREEN_WIDTH + local_center_x]:
            sh = font.render(state_text, True, (180, 100, 0))
            screen.blit(sh, sh.get_rect(center=(scr_cx + 2, user_screen_y - 118)))
            screen.blit(text_surf, text_surf.get_rect(center=(scr_cx, user_screen_y - 120)))

    pygame.display.flip()


def draw_score_screen(screen):
    screen.fill(SCORE_BG)

    # 背景に薄い星
    for sx, sy, sz in _STARS:
        pts = _star_polygon(sx, sy, sz // 2, max(sz // 5, 1))
        pygame.draw.polygon(screen, (80, 40, 110), pts)

    scores = load_scores()
    cx = TOTAL_WIDTH // 2

    # ── タイトル ──
    big_font = pygame.font.SysFont("arial", 42, bold=True)
    for ox, oy in [(-2, 2), (2, 2)]:
        sh = big_font.render("TOP 5 SCORES", True, SCORE_SHD)
        screen.blit(sh, sh.get_rect(center=(cx + ox, 60 + oy)))
    title_surf = big_font.render("TOP 5 SCORES", True, SCORE_COL)
    screen.blit(title_surf, title_surf.get_rect(center=(cx, 60)))

    # ── ヘッダー行 ──
    hdr_font = pygame.font.SysFont("arial", 20, bold=True)
    headers = ["Rank", "Date", "Error X(m)", "Error Z(m)", "Total Error"]
    col_x   = [60, 190, 440, 590, 730]
    for hx, ht in zip(col_x, headers):
        hs = hdr_font.render(ht, True, (200, 200, 255))
        screen.blit(hs, (hx, 128))
    pygame.draw.line(screen, (150, 100, 200), (40, 152), (TOTAL_WIDTH - 40, 152), 1)

    # ── スコア行 ──
    row_font = pygame.font.SysFont("arial", 22)
    rank_colors = [(255, 215, 0), (192, 192, 192), (205, 127, 50)]

    for i, entry in enumerate(scores):
        y = 165 + i * 58
        row_color = rank_colors[i] if i < 3 else (220, 220, 220)

        # 行背景（半透明風）
        bg_surf = pygame.Surface((TOTAL_WIDTH - 80, 48), pygame.SRCALPHA)
        bg_surf.fill((255, 255, 255, 50 if i % 2 == 0 else 20))
        screen.blit(bg_surf, (40, y - 4))

        vals = [
            f"#{i + 1}",
            entry["date"][5:16],       # MM-DD HH:MM
            f"{entry['error_x']:+.3f}",
            f"{entry['error_z']:+.3f}",
            f"{entry['total_error']:.3f}",
        ]
        for hx, val in zip(col_x, vals):
            vs = row_font.render(val, True, row_color)
            screen.blit(vs, (hx, y))

    if not scores:
        empty = row_font.render("No scores yet. Play a trial first!", True, (200, 180, 255))
        screen.blit(empty, empty.get_rect(center=(cx, 290)))

    # ── 黒猫（右下）──
    draw_cat(screen, TOTAL_WIDTH - 80, SCREEN_HEIGHT - 10,
             message="ランキングにゃ！", black=True)

    # ── 操作説明 ──
    hint_font = pygame.font.SysFont("arial", 19)
    hint = hint_font.render("Press  S  or  ESC  to return", True, (180, 160, 220))
    screen.blit(hint, hint.get_rect(center=(cx, SCREEN_HEIGHT - 28)))

    pygame.display.flip()


# ==============================================================================
# 【試行ロジック】
# ==============================================================================
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
    """戻り値: (continue_flag, error_x | None, error_z | None)"""
    speed = CONSTANT_SPEED
    side = random.choice([-1, 1])
    dist_val = random.uniform(1.0, 3.0)
    obj_x = round(side * dist_val, 2)

    current_z = START_Z
    pass_time_needed = START_Z / speed

    screen = pygame.display.get_surface()
    local_center_x = SINGLE_SCREEN_WIDTH // 2

    source.pause()
    run_countdown(screen, font, ctx, beep_low, beep_high, obj_x, START_Z)

    print(f"\n[試行開始] 速度:{speed}m/s, 位置X:{obj_x}m")
    start_time = time.time()
    clicked = False
    result_error_x = None
    result_error_z = None

    source.play()

    while current_z >= END_Z:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False, None, None

            if event.type == pygame.MOUSEBUTTONDOWN and not clicked:
                mouse_x, _ = event.pos
                rel_x_px = mouse_x - SINGLE_SCREEN_WIDTH
                if rel_x_px < 0:
                    rel_x_px = mouse_x
                diff_px = rel_x_px - local_center_x
                clicked_x_meter = diff_px / SCALE

                click_time = time.time() - start_time
                lag = click_time - pass_time_needed
                error_z = current_z
                error_x = clicked_x_meter - obj_x

                result_error_x = error_x
                result_error_z = error_z

                result_data = [
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    SIM_FPS, speed, obj_x,
                    round(clicked_x_meter, 3),
                    round(error_x, 3),
                    round(error_z, 3),
                    round(lag, 3),
                ]
                save_to_csv(result_data)
                print(f"  -> 計測: 正解X={obj_x}m, 回答X={clicked_x_meter:.2f}m")
                clicked = True
                source.pause()

        if not clicked:
            source.position.value = (obj_x, 0.0, current_z)
            d = np.sqrt(obj_x ** 2 + current_z ** 2)
            p_factor = calculate_pitch_factor(d)
            try:
                generator.pitch_bend.value = p_factor
            except Exception:
                generator.pitch_bend.value = 1.0

        status_text = "CLICKED!" if clicked else ""
        draw_dual_screen(screen, font, status_text, obj_x, current_z, show_ball_on_admin=True)

        current_z -= speed * (1.0 / SIM_FPS)
        time.sleep(1.0 / SIM_FPS)

    source.pause()
    return True, result_error_x, result_error_z


# ==============================================================================
# 【メインループ】
# ==============================================================================
def main():
    pygame.init()
    pygame.font.init()
    font = pygame.font.SysFont("arial", 36, bold=True)

    screen = pygame.display.set_mode((TOTAL_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption(f"AUDIO CATCHER - {CSV_FILENAME}")

    init_stars()

    synthizer.initialize()
    try:
        ctx = synthizer.Context()
        ctx.default_panner_strategy.value = synthizer.PannerStrategy.HRTF

        beep_low  = create_beep_buffer(ctx, 440.0)
        beep_high = create_beep_buffer(ctx, 880.0)

        sound_path = os.path.join(get_project_root(), "sound_files", "2000Hz.wav")
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
        show_scores = False

        while running_main:
            if show_scores:
                draw_score_screen(screen)
            else:
                draw_dual_screen(screen, font,
                                 "SPACE: Start    S: Scores",
                                 0, START_Z, show_ball_on_admin=False)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running_main = False
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        show_scores = not show_scores
                    elif event.key == pygame.K_ESCAPE:
                        if show_scores:
                            show_scores = False
                        else:
                            running_main = False
                    elif event.key == pygame.K_SPACE and not show_scores:
                        cont, ex, ez = run_single_trial(
                            ctx, generator, source, beep_low, beep_high, font
                        )
                        if not cont:
                            running_main = False
                        elif ex is not None and ez is not None:
                            save_score(ex, ez)
                        time.sleep(0.5)

            time.sleep(0.1)

    finally:
        pygame.quit()
        synthizer.shutdown()
        print(f"\n実験終了。データは {CSV_FILENAME} に保存されました。")


if __name__ == "__main__":
    main()
