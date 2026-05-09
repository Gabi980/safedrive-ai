from pathlib import Path
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = Path("models/face_landmarker.task")
CAMERA_INDEX = 0
PREFERRED_CAMERA_WIDTH = 1280
PREFERRED_CAMERA_HEIGHT = 720
EYE_CLOSED_EAR_THRESHOLD = 0.20
DROWSINESS_SECONDS_THRESHOLD = 2.0
YAWN_MAR_THRESHOLD = 0.60

LEFT_EYE_POINTS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_POINTS = [362, 263, 387, 386, 385, 373, 374, 380]
LEFT_EAR_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EAR_POINTS = [362, 385, 387, 263, 373, 380]
MOUTH_POINTS = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317]
MOUTH_MAR_POINTS = [61, 81, 13, 291, 311, 14]
HEAD_POSE_POINTS = [1, 10, 152]


def open_camera():
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, PREFERRED_CAMERA_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, PREFERRED_CAMERA_HEIGHT)
    return cap


def print_camera_resolution(cap):
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")


def draw_landmark_points(frame, landmarks, point_indexes, color, radius=3):
    height, width, _ = frame.shape

    for index in point_indexes:
        landmark = landmarks[index]
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), radius, color, -1)


def draw_all_landmarks(frame, landmarks):
    height, width, _ = frame.shape

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        cv2.circle(frame, (x, y), 1, (180, 180, 180), -1)


def get_pixel_point(frame, landmark):
    height, width, _ = frame.shape
    return int(landmark.x * width), int(landmark.y * height)


def euclidean_distance(point_a, point_b):
    x1, y1 = point_a
    x2, y2 = point_b
    return ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5


def calculate_ear(frame, landmarks, eye_points):
    p1, p2, p3, p4, p5, p6 = [
        get_pixel_point(frame, landmarks[index]) for index in eye_points
    ]

    vertical_distance_1 = euclidean_distance(p2, p6)
    vertical_distance_2 = euclidean_distance(p3, p5)
    horizontal_distance = euclidean_distance(p1, p4)

    if horizontal_distance == 0:
        return 0.0

    return (vertical_distance_1 + vertical_distance_2) / (2.0 * horizontal_distance)


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


def get_eye_status(average_ear, closed_start_time):
    if average_ear >= EYE_CLOSED_EAR_THRESHOLD:
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


def create_face_landmarker():
    base_options = python.BaseOptions(model_asset_path=str(MODEL_PATH))
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return vision.FaceLandmarker.create_from_options(options)


def main():
    if not MODEL_PATH.exists():
        print(f"Missing model: {MODEL_PATH}")
        return

    cap = open_camera()

    if not cap.isOpened():
        print("Camera unavailable.")
        return

    print_camera_resolution(cap)
    landmarker = create_face_landmarker()
    print("Face Mesh started. Press Q or Esc to exit.")
    closed_start_time = None

    while True:
        success, frame = cap.read()

        if not success:
            print("Frame read failed.")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp_ms = int(time.time() * 1000)
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.face_landmarks:
            face_landmarks = result.face_landmarks[0]

            # draw_all_landmarks(frame, face_landmarks)
            draw_landmark_points(frame, face_landmarks,
                                 LEFT_EYE_POINTS, (0, 255, 0))
            draw_landmark_points(frame, face_landmarks,
                                 RIGHT_EYE_POINTS, (0, 255, 0))
            draw_landmark_points(frame, face_landmarks,
                                 MOUTH_POINTS, (255, 0, 255))
            draw_landmark_points(frame, face_landmarks,
                                 HEAD_POSE_POINTS, (0, 255, 255), radius=4)

            left_ear = calculate_ear(frame, face_landmarks, LEFT_EAR_POINTS)
            right_ear = calculate_ear(frame, face_landmarks, RIGHT_EAR_POINTS)
            average_ear = (left_ear + right_ear) / 2.0
            eye_status, eye_status_color, closed_start_time = get_eye_status(
                average_ear,
                closed_start_time,
            )
            mar = calculate_mar(frame, face_landmarks, MOUTH_MAR_POINTS)
            mouth_status, mouth_status_color = get_mouth_status(mar)

            status_text = "Face detected"
            status_color = (0, 255, 0)

            cv2.putText(
                frame,
                f"EAR: {average_ear:.2f}",
                (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                eye_status,
                (20, 105),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                eye_status_color,
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"MAR: {mar:.2f}",
                (20, 140),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                mouth_status,
                (20, 175),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                mouth_status_color,
                2,
                cv2.LINE_AA,
            )
        else:
            status_text = "No face detected"
            status_color = (0, 0, 255)
            closed_start_time = None

        cv2.putText(
            frame,
            status_text,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            status_color,
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("SafeDrive AI - Face Mesh", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    landmarker.close()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
