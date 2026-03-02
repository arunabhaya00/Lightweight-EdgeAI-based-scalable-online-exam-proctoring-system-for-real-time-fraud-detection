
import os
import cv2
import numpy as np
import argparse
import warnings
import time

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name
warnings.filterwarnings('ignore')


SAMPLE_IMAGE_PATH = "./images/sample/"


def check_image(image):
    height, width, channel = image.shape
    if width/height != 3/4:
        print("Image is not appropriate!!!\nHeight/Width should be 4/3.")
        return False
    else:
        return True


def test(image_name, model_dir, device_id):
    import cv2
    import numpy as np
    import os

    from src.anti_spoof_predict import AntiSpoofPredict
    from src.generate_patches import CropImage
    from src.utility import parse_model_name

    anti_spoof = AntiSpoofPredict(device_id)
    image_cropper = CropImage()

    cap = cv2.VideoCapture(device_id, cv2.CAP_DSHOW)

    print("Camera opened:", cap.isOpened())

    if not cap.isOpened():
        input("Camera not opened. Press Enter to exit...")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Cannot read frame")
            break

        # ALWAYS show frame
        image_bbox = anti_spoof.get_bbox(frame)

        if image_bbox is not None:
            prediction = np.zeros((1, 3))

            for model_name in os.listdir(model_dir):
                h_input, w_input, model_type, scale = parse_model_name(model_name)
                param = {
                    "org_img": frame,
                    "bbox": image_bbox,
                    "scale": scale,
                    "out_w": w_input,
                    "out_h": h_input,
                    "crop": True,
                }

                img = image_cropper.crop(**param)
                prediction += anti_spoof.predict(
                    img, os.path.join(model_dir, model_name)
                )

            label = np.argmax(prediction)

            if label == 1:
                text = "REAL"
                color = (0, 255, 0)
            else:
                text = "FAKE"
                color = (0, 0, 255)

            x, y, w, h = image_bbox
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, text, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        else:
            cv2.putText(frame, "NO FACE DETECTED", (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Anti-Spoofing Test", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    MODEL_DIR = "resources/anti_spoof_models"
    DEVICE_ID = 0   # change to 1 if needed

    print("Running anti-spoofing webcam test...")
    test(None, MODEL_DIR, DEVICE_ID)
