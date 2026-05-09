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

LEFT_EYE_POINTS = [33, 133, 160, 159, 158, 144, 145, 153]
RIGHT_EYE_POINTS = [362, 263, 387, 386, 385, 373, 374, 380]
MOUTH_POINTS = [61, 291, 13, 14, 78, 308, 82, 312, 87, 317]
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

            status_text = "Face detected"
            status_color = (0, 255, 0)
        else:
            status_text = "No face detected"
            status_color = (0, 0, 255)

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
