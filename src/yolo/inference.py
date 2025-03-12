"""
Inference with YOLO on images
"""

import time
import argparse
import cv2

from ultralytics import YOLO

def draw_bounding_boxes(frame, boxes, confidences, classes, class_names):
    for box, conf, cls in zip(boxes, confidences, classes):
        x1, y1, x2, y2 = map(int, box)
        label = f'{class_names[int(cls)]}: {conf:.2f}'

        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw label
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def main(opts: argparse.Namespace) -> None:
    """main

    Parameters
    ----------
    opts : `argparse.Namespace`
        Command-line options
    """

    model = YOLO(model=opts.model, verbose=True)  # load a pretrained model
    class_names = model.names  # Get class names from the model

    frame_skip = 2  # Skip every 2 frames to improve performance
    frame_count = 0
    
    # webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while 1:
        _ = cap.grab() # discard one frame to circumvent capture buffering
        _, frame = cap.read()
        frame_count += 1
        if frame_count % frame_skip != 0:  # Skip frames to boost performance
            continue

        try:
            start_time = time.monotonic()

            results = model(frame)

            # Extract results
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confidences = results[0].boxes.conf.cpu().numpy()
            classes = results[0].boxes.cls.cpu().numpy()

            if len(boxes) > 0:
                draw_bounding_boxes(frame, boxes, confidences, classes, class_names)

            # Calculate FPS
            fps = 1.0 / (time.monotonic() - start_time)
            cv2.putText(frame, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        except Exception as e:
            print(f"Error during inference: {e}")

        cv2.imshow("YOLO Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    

if __name__ == "__main__":
    # command line arguments
    args = argparse.ArgumentParser()
    args.add_argument(
        "--image", "-I", default="", help="name of image to run inference on",
    )
    args.add_argument(
        "--model",
        default="",
        help="path to YOLO model",
    )
    opts = args.parse_args()

    # main entry point
    main(opts)
