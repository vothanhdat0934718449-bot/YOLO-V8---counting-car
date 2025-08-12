# Sử dụng YOLOv8 từ ultralytics như trong source
from ultralytics import YOLO
import cv2
import cvzone
import math
# Import DeepSort từ deepsort-realtime như trong source
from deep_sort_realtime.deepsort_tracker import DeepSort
import numpy as np

cap = cv2.VideoCapture("../Videos/cars.mp4") # For Video
model = YOLO("../Yolo-Weights/yolov8l.pt") # Sử dụng YOLOv8n model

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask = cv2.imread("../mask.png")

#Tracking
tracker = DeepSort(max_age=1) # Khởi tạo DeepSort tracker

limits = [400, 297, 673, 297]

totalCount = []

while True:
    success, img = cap.read()
    if not success: # Thêm kiểm tra nếu đọc frame thất bại như trong source
        print("Đọc frame thất bại")
        break # Thoát vòng lặp nếu không đọc được frame

    imgRegion = cv2.bitwise_and(img, mask)
    imgGraphics = cv2.imread("../graphics.png", cv2.IMREAD_UNCHANGED)
    img = cvzone.overlayPNG(img, imgGraphics, (0, 0))

    results = model(imgRegion, stream=True)

    detect = []

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box ở định dạng [x1, y1, x2, y2]
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            w, h = x2 - x1, y2 - y1

            # Lấy Confidence và Class Name/ID
            conf = math.ceil((box.conf * 100)) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            # Có thể điều chỉnh các lớp ở đây
            if currentClass in ["car", "truck", "bus"] and conf > 0.3:
                detect.append(([x1, y1, x2 - x1, y2 - y1], conf, cls))  # Dạng đúng cho DeepSort

    tracks = tracker.update_tracks(detect, frame=img)

    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 0, 255), 5)

    for track in tracks: # Vòng lặp mới cho DeepSort
        if track.is_confirmed():
            # Lấy ID track
            track_id = track.track_id

            # Lấy toạ độ bbox ở định dạng ltrb (x1, y1, x2, y2)
            ltrb = track.to_ltrb()
            x1, y1, x2, y2 = map(int, ltrb) # Chuyển sang kiểu integer
            # Kiểm tra nếu bbox nằm trong vùng khung hình
            w, h = x2 - x1, y2 - y1

            # Lấy class_id của detection gắn với track này (DeepSort lưu class_id)
            class_id = track.get_det_class()
            cvzone.cornerRect(img, (x1, y1, w, h), l=9, rt=2, colorR=(255, 0, 255))

            display_text = f"#: {track_id} {classNames[class_id]}"
            cvzone.putTextRect(img, display_text, (max(0, x1), max(35, y1)),
                               scale=2, thickness=3, offset=10)

            # Tính toán tâm bounding box từ bbox theo dõi [15]
            cx, cy = x1 + w // 2, y1 + h // 2
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

            # Kiểm tra đường kẻ và cập nhật biến đếm [16]
            if limits[0] < cx < limits[2] and limits[1] - 15 < cy < limits[1] + 15:
                if track_id not in totalCount: # Sử dụng ID track để đếm duy nhất
                    totalCount.append(track_id)
                    cv2.line(img, (limits[0], limits[1]), (limits[2], limits[3]), (0, 255, 0), 5) # Đổi màu đường kẻ khi có vật thể đi qua [16]

    # Hiển thị số lần đếm
    cv2.putText(img, str(len(totalCount)), (255, 100), cv2.FONT_HERSHEY_PLAIN, 5, (50, 50, 255), 8)
    # Hiển thị frame kết quả [17]
    cv2.imshow("Image", img)
    # cv2.imshow("ImageRegion", imgRegion) [17]
    cv2.waitKey(1)



