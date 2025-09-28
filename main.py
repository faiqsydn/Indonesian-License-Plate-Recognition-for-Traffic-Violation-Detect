import os
path = "D:/School Files/PBL 3/licensePlate/licensePlate/"
os.chdir(path)
from datetime import datetime

import numpy as np
from ultralytics import YOLO
import cv2
import pandas as pd

from util import *
from sort import *

from LogoDetection2 import detect_logos


results = {}
mot_tracker = Sort()
#load models
coco_model = YOLO('yolov8x.pt')
license_plate_detector = YOLO('D:/School Files/PBL 3/licensePlate/licensePlate/licensePlateRetraining/runs/detect/train8/weights/best.pt')
# car_shape_detector = YOLO('something')

#load video
cap = cv2.VideoCapture(r'D:\School Files\PBL 3\licensePlate\licensePlate\newer.mp4')

# Define car shape labels
car_shape_labels = {
    0: "Cap",
    1: "Convertible",
    2: "Coupe",
    3: "Hatchback",
    4: "Minivan",
    5: "Other",
    6: "Sedan",
    7: "SUV",
    8: "Van",
    9: "Wagon"
}

vehicles = [2]
car_dict = {}

# read frames
frame_nmr = -1
ret = True
while ret:
    frame_nmr += 1
    ret, frame = cap.read()
    if ret:
        results[frame_nmr] = {}
        # detect vehicles
        detections = coco_model(frame)[0]
        detections_ = []
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in vehicles:
                detections_.append([x1, y1, x2, y2, score])

        # track vehicles
        if len(detections_) > 0:
            track_ids = mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = np.empty((0, 5))  # Handle case with no detections

        # Filter track_ids to keep only those with detections in the current frame
        current_frame_track_ids = []
        for track_id in track_ids:
            xcar1, ycar1, xcar2, ycar2, car_id = track_id
            for detection in detections_:
                dx1, dy1, dx2, dy2, dscore = detection
                if (dx1 == xcar1 and dy1 == ycar1 and dx2 == xcar2 and dy2 == ycar2):
                    current_frame_track_ids.append(track_id)
                    break

        # detect license plates
        license_plates = license_plate_detector(frame)[0]
        print(f'AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA {frame_nmr}')
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate

            # assign license plate to car 
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            car_crop = frame[int(ycar1):int(ycar2), int(xcar1): int(xcar2), :]
            car_dict[car_id] = 0

            if car_id != -1:

                # crop license plate
                license_plate_crop = frame[int(y1):int(y2), int(x1): int(x2), :]
                # process license plate
                license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 120, 255, cv2.THRESH_BINARY)
                license_plate_crop_thresh = cv2.bitwise_not(license_plate_crop_thresh)
                # read license plate number
                license_plate_text, license_plate_text_score = read_license_plate(license_plate_crop_thresh, car_crop, frame_nmr)
                # check violation
                df = pd.read_csv("register.csv")
                database_license_plate = df[df["licensePlate"] == license_plate_text]
                violation = "True"
                violation_type = "NoReg"
                if not database_license_plate.empty:
                    current_date = int(datetime.now().day)
                    last_number = 0
                    for char in license_plate_text:
                        if char.isdigit():
                            last_number = int(char)
                    if current_date % 2 == 0:
                        if int(last_number) % 2 == 0:
                            violation = "False"
                    else: 
                        if last_number % 2 == 1:
                            violation = "False"
                    violation_type = "OddEven"
                    

                if license_plate_text is not None:
                    if car_dict[car_id] < license_plate_text_score:
                        car_dict[car_id] = license_plate_text_score
                        cv2.imwrite(f"D:/School Files/PBL 3/licensePlate/licensePlate/detectedCars/{int(car_id)}.jpg", car_crop)
                        cv2.imwrite(f"D:/School Files/PBL 3/licensePlate/licensePlate/detectedLicensePlate/{int(car_id)}.jpg", license_plate_crop_thresh)
                        cv2.imwrite(f"D:/School Files/PBL 3/licensePlate/licensePlate/detectedLicensePlate/{int(car_id)}_raw.jpg", license_plate_crop_gray)

                    results[frame_nmr][car_id] = {
                        'car': {
                            'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text, 
                            'bbox_score': score,
                            'text_score': license_plate_text_score},
                        'violation': violation,
                        'violation_type' : violation_type,
                        'confirmed': "False"
                        }
                    
        # # Detect car shapes in the current frame's cars
        # for track_id in current_frame_track_ids:
        #     xcar1, ycar1, xcar2, ycar2, car_id = track_id

        #     # Crop car region
        #     car_crop = frame[int(ycar1):int(ycar2), int(xcar1):int(xcar2), :]

        #     # Detect car shape in the cropped region
        #     car_shapes = car_shape_detector(car_crop)[0]

        #     # Process car shapes (example, you can modify based on your requirement)
        #     for car_shape in car_shapes.boxes.data.tolist():
        #         sx1, sy1, sx2, sy2, sscore, sclass_id = car_shape
        #         car_shape_label = car_shape_labels.get(int(sclass_id), 'Unknown')
        #         if car_id in results[frame_nmr]:
        #             results[frame_nmr][car_id]['car_shape'] = {'bbox': [sx1, sy1, sx2, sy2],
        #                                                        'bbox_score': sscore,
        #                                                        'class_id': sclass_id,
        #                                                        'label': car_shape_label}
        #         else:
        #             results[frame_nmr][car_id] = {'car_shape': {'bbox': [sx1, sy1, sx2, sy2],
        #                                                         'bbox_score': sscore,
        #                                                         'class_id': sclass_id,
        #                                                         'label': car_shape_label}}
                    
        # # Detect logos in the cropped car region
        #     _, img_encoded = cv2.imencode('.jpg', car_crop)
        #     content = img_encoded.tobytes()
        #     logos = detect_logos(content)

        #     # Process logos
        #     for logo in logos:
        #         vertices = logo['boundingPoly']['vertices']
        #         pts = [(vertex.get('x', 0), vertex.get('y', 0)) for vertex in vertices]
        #         pts = pts + [pts[0]]
        #         pts = [(int(x), int(y)) for x, y in pts]
        #         logo_description = logo['description']
        #         if car_id in results[frame_nmr]:
        #             if 'logos' not in results[frame_nmr][car_id]:
        #                 results[frame_nmr][car_id]['logos'] = []
        #             results[frame_nmr][car_id]['logos'].append({'description': logo_description,
        #                                                         'vertices': pts})

# write results
write_csv(results, './detected.csv', './violation.csv')