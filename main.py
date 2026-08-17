import cv2
from ultralytics import YOLO


def main():
    # Load the small YOLO model.
    # It will download automatically the first time.
    model = YOLO("yolo26n.pt")

    # Open the default webcam
    camera = cv2.VideoCapture(0)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    print("Webcam started.")
    print("Press Q to quit.")

    while True:
        success, frame = camera.read()

        if not success:
            print("ERROR: Could not read frame.")
            break

        # Detect objects in the current frame
        results = model.predict(
            source=frame,
            imgsz=320,
            conf=0.35,
            verbose=False
        )

        # Get the first result
        result = results[0]

        # Draw bounding boxes and labels
        annotated_frame = result.plot()

        # Show the camera
        cv2.imshow(
            "Motion Detector - Stage 1",
            annotated_frame
        )

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()