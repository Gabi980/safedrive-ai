import cv2


def main():
    camera_index = 0
    cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        print("Camera unavailable.")
        return

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
