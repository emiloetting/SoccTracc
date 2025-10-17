from ultralytics import YOLO, settings
import numpy as np
import os


# Disable analytics and crash reporting
settings.update({"sync": False})
class Detector:
    def __init__(self, model:str, cwd: str):
        self.name = "Detector"  # Do not change the name of the module as otherwise recording replay would break!
        self.model = YOLO(os.path.join(cwd, "pipeline","detector_models", model))

    def start(self, data):
        pass

    def stop(self, data):
        pass

    def step(self, data):
        """Runs the YOLO model on the input image to detect players.
        Args:
            data (dict): Dictionary with keys:
                image (np.ndarray): BGR input frame.
        Returns:
            dict: Contains:
                detections (np.ndarray): Array of bounding boxes in xywh format.
                classes (np.ndarray): Array of class indices for each detection.
        """
        results = self.model(data["image"], verbose=False)
        xywh = np.asarray(results[0].boxes.xywh)
        classes = np.asarray(results[0].boxes.cls)

        return {"detections": xywh, "classes": classes}
