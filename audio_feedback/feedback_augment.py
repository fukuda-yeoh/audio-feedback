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
