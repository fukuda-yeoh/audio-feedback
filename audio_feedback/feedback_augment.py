# 音の出力制御の関数


# Function to calculate volume based on distance (forward-backward)
def calculate_volume(
    distance, reference_distance=1.0, max_volume=200.0, min_volume=20.0
):
    if distance < reference_distance:
        distance = reference_distance
    volume = max_volume / (distance / reference_distance) ** 2
    return max(min(volume, max_volume), min_volume)


# 音の制御（x軸）の強調
def calculate_pan(x_position, img_width, pan_strength=2.0):
    pan = (x_position - img_width / 2) / (img_width / 2)
    pan *= pan_strength  # パンニングを強調
    return max(min(pan, 1), -1)  # Value between -1 (left) and 1 (right)


# 音の制御（y軸）
def calculate_pitch(y_position, img_height):
    pitch = 1.0 + (y_position / img_height)  # Pitch scales from 1.0 to 2.0
    return pitch


def calculate_point_source_gain(
    distance, distance_min=1.0, distance_max=4.0, gain_max=1.0, gain_min=0.0
):
    """
    点音源モデルに基づくゲイン計算
    """
    if distance < distance_min:
        return gain_max
    if distance > distance_max:
        return gain_min

    # 距離の逆比例に基づいてゲインを計算
    gain = gain_max / distance

    # ゲインが最小値を下回らないように制限
    return max(gain_min, gain)


def calculate_realistic_gain(
    distance, distance_min, distance_max, gain_max=1.0, gain_min=0.0
):
    """
    音の物理的な減衰モデル（逆二乗則）
    """
    if distance < distance_min:
        return gain_max
    if distance > distance_max:
        return gain_min

    # 逆二乗則に基づいてゲインを計算
    gain = gain_max / (distance**2)

    # ゲインが最小値を下回らないように制限
    return max(gain_min, gain)


def calculate_gain(
    distance, distance_min=1.0, distance_max=4.0, gain_max=1.0, gain_min=0.0
):
    # 距離の範囲を超えた場合は音量を最小値に設定
    if distance > distance_max:
        return gain_min
    if distance < distance_min:
        return gain_max

    # 減衰率を計算
    k = (gain_max - gain_min) / (distance_max - distance_min)

    # 音量を計算
    gain = gain_max - k * (distance - distance_min)
    return max(gain_min, gain)  # 音量が負にならないように調整


def LINEAR_pitch(distance, d_ref, d_max):
    clamp = max(d_ref, min(distance, d_max))
    pitch = 2 - 1.5 * (clamp - d_ref) / (d_max - d_ref)
    return pitch


def EXPONENTIAL_pitch(distance, d_ref):
    pitch = 2 * (max(d_ref, distance) / d_ref) ** -0.68
    return max(pitch, 0.5)


def INVERSE_pitch(
    distance,
    d_ref,
):
    pitch = d_ref / (d_ref + 0.5 * (max(d_ref, distance) - d_ref))
    return max(pitch, 0.5)
