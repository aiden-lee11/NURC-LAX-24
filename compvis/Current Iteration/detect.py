import numpy as np
import cv2 as cv


class BinaryMotionDetector:
    def __init__(self, hsv, deltas, frame_delay=15, min_mean=0.1):
        self.hsv = hsv
        self.deltas = deltas
        self.frame_delay = frame_delay
        self.min_mean = min_mean

        self._update_thresholds()
        self.kernel = np.ones((3, 3), np.uint8)
        self.past_mask_queue = []
        self.past_mask = 0

    def set_hsv(self, hsv):
        self.hsv = hsv

    def set_deltas(self, deltas):
        self.hsv_deltas = deltas

    def set_frame_delay(self, frame_delay):
        self.frame_delay = frame_delay

    def _update_thresholds(self):
        h, s, v = self.hsv
        del_h, del_s, del_v = self.deltas

        self.lower_threshold = (h - del_h, s - del_s, v - del_v)
        self.higher_threshold = (h + del_h, s + del_s, v + del_v)

    def detect(self, frame):
        frame = cv.GaussianBlur(frame, (3, 3), 0)
        frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV_FULL)

        mask = cv.inRange(frame_hsv, self.lower_threshold, self.higher_threshold)
        motion_mask = mask - self.past_mask
        motion_mask[motion_mask < 128] = 0  # Strange case where elements are 1
        motion_mask = cv.erode(motion_mask, self.kernel, iterations=5)
        motion_mask = cv.dilate(motion_mask, self.kernel, iterations=1)

        detected_y, detected_x = np.nonzero(motion_mask)
        if len(detected_y) == 0 or len(detected_x) == 0:
            detected_y = [0]
            detected_x = [0]
        y_median = int(np.median(detected_y))
        x_median = int(np.median(detected_x))

        # mask_frame = cv.cvtColor(motion_mask, cv.COLOR_GRAY2BGR)

        self.past_mask_queue.append(mask)
        if self.frame_delay > 1:
            self.frame_delay -= 1
        else:
            self.past_mask = self.past_mask_queue.pop(0)

        ret = True
        if np.mean(motion_mask) < self.min_mean:
            ret = False
            y_median = None
            x_median = None

        return ret, y_median, x_median


if __name__ == "__main__":
    import camera

    hsv = [12, 175, 225]
    deltas = [16, 80, 120]

    cam_1 = camera.Camera("Main Camera Name")
    detector = BinaryMotionDetector(hsv, deltas)

    camera.assign_captures([cam_1])
    while True:
        ret, frame = cam_1.get_frame()
        if not ret:
            print("Frame not received successfully")
            break

        ret, y_median, x_median = detector.detect(frame)
        if ret:
            cv.circle(frame, (x_median, y_median), 8, (0, 0, 255), -1)

        cv.imshow("Motion detection", frame)
        if cv.waitKey(1) == ord("q"):
            break
        
    cam_1.release()
    cv.destroyAllWindows()