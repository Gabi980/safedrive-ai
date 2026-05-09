import cv2


CAMERA_INDEX = 0
PREFERRED_CAMERA_WIDTH = 1280
PREFERRED_CAMERA_HEIGHT = 720


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


def main():
    cap = open_camera()

    if not cap.isOpened():
        print("Camera unavailable.")
        return

    print_camera_resolution(cap)
    print("Camera started. Press Q or Esc to exit.")

    while True:
        success, frame = cap.read()

        if not success:
            print("Frame read failed.")
            break

        frame = cv2.flip(frame, 1)

        cv2.imshow("SafeDrive AI - Camera Live", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
