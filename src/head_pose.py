import cv2
import numpy as np

from config import (
    HEAD_DOWN_PITCH_THRESHOLD,
    HEAD_TILT_ROLL_THRESHOLD,
    HEAD_UP_PITCH_THRESHOLD,
    SIDE_LOOK_YAW_THRESHOLD,
)
from landmark_indexes import HEAD_POSE_MODEL_POINTS, HEAD_POSE_POINTS


def get_pixel_point(frame, landmark):
    height, width, _ = frame.shape
    return int(landmark.x * width), int(landmark.y * height)


def estimate_head_pose(frame, landmarks):
    height, width, _ = frame.shape
    image_points = np.array(
        [get_pixel_point(frame, landmarks[index]) for index in HEAD_POSE_POINTS],
        dtype=np.float64,
    )

    focal_length = width
    center = (width / 2, height / 2)
    camera_matrix = np.array(
        [
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    distortion_coefficients = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, translation_vector = cv2.solvePnP(
        HEAD_POSE_MODEL_POINTS,
        image_points,
        camera_matrix,
        distortion_coefficients,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )

    if not success:
        return None

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    projection_matrix = np.hstack((rotation_matrix, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(projection_matrix)

    pitch, yaw, roll = euler_angles.flatten()

    return (
        normalize_angle(float(pitch)),
        normalize_angle(float(yaw)),
        normalize_angle(float(roll)),
    )


def normalize_angle(angle):
    if angle > 90:
        return angle - 180

    if angle < -90:
        return angle + 180

    return angle


def get_head_status(head_pose):
    if head_pose is None:
        return "Head pose unavailable", (0, 0, 255)

    pitch, yaw, roll = head_pose

    if abs(yaw) > SIDE_LOOK_YAW_THRESHOLD:
        return "Looking sideways", (0, 165, 255)

    if pitch > HEAD_DOWN_PITCH_THRESHOLD:
        return "Head down", (0, 0, 255)

    if pitch < HEAD_UP_PITCH_THRESHOLD:
        return "Head up", (0, 165, 255)

    if abs(roll) > HEAD_TILT_ROLL_THRESHOLD:
        return "Head tilted", (0, 165, 255)

    return "Head normal", (0, 255, 0)
