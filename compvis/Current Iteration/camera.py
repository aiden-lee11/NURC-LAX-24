import cv2 as cv
import time
import numpy as np
from masks import binary_centroid, get_hsv_ranges
from detect import BinaryMotionDetector
from timers import timers
#from threading import Thread


class Cam:
    """
    Camera class representing a video camera.

    Attributes:
    - cap: VideoCapture object for camera.
    - frame: Current frame from the camera.
    - hsv_value: HSV value range for color detection. In form (hsv_lower, hsv_upper) with tuples of each
    - name: Name of the camera window.
    - camID: Camera ID.
    - x: X-coordinate of detected object.
    - y: Y-coordinate of detected object.
    - R: Radius of detected object.
    - fps: Frames per second.
    """

    def __init__(self, name, camID=None):
        self.cap = None
        self.frame = None
        self.hsv_value = None
        self.name = name
        self.camID = camID
        self.x = None
        self.y = None
        self.R = 10
        self.fps = None

        # cam_matrix hold [[fx, 0, ox], 
        #                   [0, fy, oy], 
        #                   [0, 0, 1]]
        self.cam_matrix = np.array(
            [[500, 0, 320], 
             [0, 500, 240], 
             [0, 0, 1]]
        )  # hard coded from camera calibration
        self.stopped = False
        #self.b_offset = b_offset

    def release_camera(self):
        self.stopped = True
        self.cap.release()
        print(f"{self.name} released!")

    def set_id(self, id):
        self.camID = id
        return id

    def get_id(self):
        return self.camID
    
    def get_name(self):
        return self.name

    def has_id(self):
        return not self.camID == -1

    def get_cap(self):
        return self.cap

    def update(self):
        while True:
            if self.stopped:
                return

            self.ret, self.frame = self.cap.read()

    def set_cap(self, cap):
        self.cap = cap
        #Thread(target=self.update, args=()).start()


    def get_frame(self):
        _, frame = self.cap.read()
        self.frame = frame
        return frame

    def ball_position(self):
        return self.x is not None and self.y is not None

    def show_circled_frame(self):
        if self.x and self.y and self.R:
            cv.circle(
                self.frame,
                (int(self.x), int(self.y)),
                int(self.R),
                (0, 255, 255),
                2,
            )

        if self.fps:
            cv.putText(
                self.frame, str(self.fps), (np.shape(self.frame)[1] - 100, 50), 0, 1, (0, 255, 0), 2
            )

        cv.imshow(self.name, self.frame)

    def run(self):
        t0 = time.time()
        with timers.timers["Get Frame"]:
            self.get_frame()
        timers.record_time("Get Frame")

        with timers.timers["Binary Centroid"]:
           binary_centroid(self)
        timers.record_time("Binary Centroid")


        with timers.timers["Show Frame"]:
            self.show_circled_frame()
        timers.record_time("Show Frame")

        self.fps = 1 / (time.time() - t0)

        return self.ball_position()

    def set_camera_id(self):
            cap = cv.VideoCapture(self.camID)
            if not cap.isOpened():
                print(f"camera ID {self.camID} is not available.")

            print("Press [0-9] to assign the capture to that camera")
            print("Press [q] to quit")

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Failed to retrieve frame from camera ID {self.camID}.")
                    break

                cv.imshow(f"camera ID {self.camID}", frame)
                key = cv.waitKey(1)
                if key in (ord(c) for c in "0123456789"):
                    camera_index = int(chr(key))
                    print(f"Assigning capture [{camera_index}] to camera [{self.camID}]")
                    self.cap = cap
                    self.name = f"Cam {camera_index}"
                    cap = cv.VideoCapture(camera_index)
                    cv.destroyAllWindows()
                    return 


                if key == ord("q"):
                    print("--> camera skipped!")
                    break

            cap.release()
            cv.destroyAllWindows()

def find_camera_ids():
    ids = []
    for cam_id in range(
        3
    ):  # Adjust the range based on the number of cameras you want to check
        cap = cv.VideoCapture(cam_id)
        ret, _ = cap.read()
        if ret:
            ids.append(cam_id)
    return ids


def camera_instantiator(cam_ids=None, static=False):
    # quick function to set up each camera
    cameras = {}
    if cam_ids is None:
        cam_ids = find_camera_ids()

    lr = 0
    pos = ["Left", "Right"]
    print("Assign 0 to the left camera and 1 to the right camera\n")
    for cam_id in cam_ids:
        cam_name = f"Cam{cam_id}"

        # cap = None fix
        test_cam = Cam(camID=cam_id, name= pos[lr])
        test_cam.set_camera_id()

        if test_cam.cap and test_cam.name:
            cameras[cam_name] = test_cam
            lr += 1
            if lr >= 2:
                break


    for camera in cameras.values():
        if static:
            hsv = [12, 175, 225]
            deltas = [16, 80, 120]

            lower_threshold = np.array([max(0, hsv[i] - deltas[i]) for i in range(3)])
            upper_threshold = np.array([min(255, hsv[i] + deltas[i]) for i in range(3)])

            camera.hsv_value = (lower_threshold, upper_threshold)
        else:
            get_hsv_ranges(camera)

    return cameras


if __name__ == "__main__":
    cameras = camera_instantiator()
    verbose = True
    while True:
        for camera in cameras.values():
            if verbose:
                print(
                    f"Does Cam{camera.get_id()} currently have a set id? {camera.has_id()}"
                )
                verbose = False
            camera.run()
        if cv.waitKey(1) == ord("q"):
            for camera in cameras.values():
                camera.release_camera()
            break
