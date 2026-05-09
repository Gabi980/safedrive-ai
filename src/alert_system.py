import time
import winsound

from config import AUDIO_ALERT_COOLDOWN_SECONDS


def get_rule_based_driver_status(eye_status, mouth_status, head_status):
    warning_signs = []

    if eye_status == "Drowsiness warning":
        warning_signs.append("eyes")

    if mouth_status == "Yawning detected":
        warning_signs.append("yawn")

    if head_status in ["Head down", "Looking sideways", "Head tilted"]:
        warning_signs.append("head")

    if len(warning_signs) >= 2:
        return "Driver drowsy", (0, 0, 255), warning_signs

    if warning_signs:
        return f"Warning: {warning_signs[0]}", (0, 165, 255), warning_signs

    return "Driver OK", (0, 255, 0), warning_signs


def calculate_drowsiness_score(eye_status, mouth_status, head_status):
    score = 0

    if eye_status == "Eyes closed":
        score += 25
    elif eye_status == "Drowsiness warning":
        score += 45

    if mouth_status == "Yawning detected":
        score += 25

    if head_status in ["Head down", "Looking sideways", "Head tilted"]:
        score += 30

    return min(score, 100)


def get_alert_level(score):
    if score < 30:
        return "Level 0: OK", (0, 255, 0), 0

    if score < 60:
        return "Level 1: Visual warning", (0, 165, 255), 1

    if score < 80:
        return "Level 2: Audio alert", (0, 80, 255), 2

    return "Level 3: Critical alert", (0, 0, 255), 3


def play_alert_sound(alert_level, last_sound_time):
    if alert_level < 2:
        return last_sound_time

    current_time = time.time()
    if current_time - last_sound_time < AUDIO_ALERT_COOLDOWN_SECONDS:
        return last_sound_time

    if alert_level == 2:
        winsound.Beep(1200, 180)
    else:
        winsound.Beep(1800, 180)
        winsound.Beep(1800, 180)

    return current_time
