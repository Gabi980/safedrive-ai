import time

from config import (
    DROWSINESS_SECONDS_THRESHOLD,
    EYE_CLOSED_EAR_THRESHOLD,
    YAWN_MAR_THRESHOLD,
)


def get_pixel_point(frame, landmark):
    height, width, _ = frame.shape
    return int(landmark.x * width), int(landmark.y * height)


def euclidean_distance(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def calculate_ear(frame, landmarks, eye_points):
    p1, p4, upper_1, lower_1, upper_2, lower_2, upper_3, lower_3 = [
        get_pixel_point(frame, landmarks[index]) for index in eye_points
    ]

    vertical_distance_1 = euclidean_distance(upper_1, lower_1)
    vertical_distance_2 = euclidean_distance(upper_2, lower_2)
    vertical_distance_3 = euclidean_distance(upper_3, lower_3)
    horizontal_distance = euclidean_distance(p1, p4)

    if horizontal_distance == 0:
        return 0.0

    average_vertical_distance = (
        vertical_distance_1 + vertical_distance_2 + vertical_distance_3
    ) / 3.0

    return average_vertical_distance / horizontal_distance


def calculate_mar(frame, landmarks, mouth_points):
    p1, p2, p3, p4, p5, p6 = [
        get_pixel_point(frame, landmarks[index]) for index in mouth_points
    ]

    vertical_distance_1 = euclidean_distance(p2, p6)
    vertical_distance_2 = euclidean_distance(p3, p5)
    horizontal_distance = euclidean_distance(p1, p4)

    if horizontal_distance == 0:
        return 0.0

    return (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)


def get_eye_status(average_ear, closed_start_time, ear_threshold=None):
    if ear_threshold is None:
        ear_threshold = EYE_CLOSED_EAR_THRESHOLD

    if average_ear >= ear_threshold:
        return "Eyes open", (0, 255, 0), None

    if closed_start_time is None:
        closed_start_time = time.time()

    closed_seconds = time.time() - closed_start_time

    if closed_seconds >= DROWSINESS_SECONDS_THRESHOLD:
        return "Drowsiness warning", (0, 0, 255), closed_start_time

    return "Eyes closed", (0, 165, 255), closed_start_time


def get_mouth_status(mar):
    if mar >= YAWN_MAR_THRESHOLD:
        return "Yawning detected", (0, 0, 255)

    return "Mouth normal", (0, 255, 0)
