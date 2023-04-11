import pyrealsense2 as rs
import numpy as np
import cv2
from ultralytics import YOLO
import cvzone
model = YOLO('yolov8n.pt')
class Detector:
    def __init__(self):
        self.classNames = ["COT"]
        self.model = YOLO('test.pt')
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []


    def detect_object(self, bgr_frame):
        res = self.model(bgr_frame, stream=True)
        self.obj_boxes = []
        self.obj_classes = []
        self.obj_centers = []
        self.obj_distances = []
        for r in res:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                w, h = x2 - x1, y2 - y1
                cls = int(box.cls[0])
                curClass = self.classNames[cls]
                if curClass == "COT":
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    self.obj_centers.append((cx, cy))
                    self.obj_boxes.append((x1, y1, w, h))
                    self.obj_classes.append(curClass)
        return self.obj_boxes, self.obj_centers, self.obj_classes

    def draw_object_info(self, bgr_frame, depth_frame):
        for i in range(len(self.obj_boxes)):
            box = self.obj_boxes[i]
            cx, cy = self.obj_centers[i]
            cls = self.obj_classes[i]
            distance = depth_frame[cy, cx]
            cvzone.cornerRect(bgr_frame, box)
            cvzone.putTextRect(bgr_frame, f"({cx},{cy}): {distance}mm", (max(box[0], 0), max(35, box[1])), 2, 2)
            cv2.circle(bgr_frame, (cx, cy), 5, (255, 0, 255), -1)
        return bgr_frame
class RealsenseCamera:
    def __init__(self):
        # Configure depth and color streams
        print("Loading Intel Realsense Camera")
        self.pipeline = rs.pipeline()

        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        # Start streaming
        self.pipeline.start(config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def get_frame_stream(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            # If there is no frame, probably camera not connected, return False
            print(
                "Error, impossible to get the frame, make sure that the Intel Realsense camera is correctly connected")
            return False, None, None

        # Apply filter to fill the Holes in the depth image
        spatial = rs.spatial_filter()
        spatial.set_option(rs.option.holes_fill, 3)
        filtered_depth = spatial.process(depth_frame)

        hole_filling = rs.hole_filling_filter()
        filled_depth = hole_filling.process(filtered_depth)

        # Create colormap to show the depth of the Objects
        colorizer = rs.colorizer()
        depth_colormap = np.asanyarray(colorizer.colorize(filled_depth).get_data())

        # Convert images to numpy arrays
        # distance = depth_frame.get_distance(int(50),int(50))
        # print("distance", distance)
        depth_image = np.asanyarray(filled_depth.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        # cv2.imshow("Colormap", depth_colormap)
        # cv2.imshow("depth img", depth_image)

        return True, color_image, depth_image

    def release(self):
        self.pipeline.stop()
        # print(depth_image)

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.10), 2)

        # Stack both images horizontally

        # images = np.hstack((color_image, depth_colormap))

rsc = RealsenseCamera()
dt = Detector()
while True:
    ret, bgr_frame, depth_frames = rsc.get_frame_stream()
    #img1 = bgr_frame
    #hsv = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
    #orange_lower = np.array([10, 100, 100])
    #orange_upper = np.array([24, 255, 255])
    #mask = cv2.inRange(hsv, orange_lower, orange_upper)
    #loc = cv2.bitwise_and(img1, img1, mask=mask)

    res = dt.detect_object(bgr_frame)
    bgr_frame = dt.draw_object_info(bgr_frame, depth_frames)

    cv2.imshow("BGR Frame", bgr_frame)
    key = cv2.waitKey(1)
    if key == 27:
        break
