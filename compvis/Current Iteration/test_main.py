import cv2
from time import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from camera import camera_instantiator
from timers import timers
from triangulation import LSLocalizer
from predict import RecursivePolynomialFit
from visualization import PointInSpace

def double_camera_test():
    """Not tested"""
    cam_1_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])
    cam_2_matrix = np.array([[500, 0, 320], [0, 500, 240], [0, 0, 1]])

    hsv = [12, 175, 225]
    deltas = [16, 80, 120]

    # hsv = [13, 179, 208]
    # deltas = [33, 33, 77]

    # first camera at origin
    T_1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    # second camera rotated pi/2 about Z at (1, 1, 0)
    T_2 = np.array(
        [
            [0, -1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    lim_x = [-1, 1]
    lim_y = [0, 2]
    lim_z = [-0.5, 0.5]
    predicted_point = [0, 0, 0]

    detection_start_time = time()
    t = time() - detection_start_time

    detected = False  # Only used to know when to print coefficients
    detected_frames = 0
    detected_frames_cap = 30
    detected_frame_threshold = 5

    cam_1 = Camera("Left Camera", cam_1_matrix)
    cam_2 = Camera("Right Camera", cam_2_matrix)

    # detector = BinaryMotionDetector(hsv, deltas, 30)  # Turn on isolation
    detector = BinaryMotionDetector(hsv, deltas, 1e10)  # Turn off isolation

    lsl = LSLocalizer([T_1, T_2])

    plotter = PointInSpace(lim_x, lim_y, lim_z)

    x_rpf = RecursivePolynomialFit(2)
    y_rpf = RecursivePolynomialFit(2)
    z_rpf = RecursivePolynomialFit(2)

    assign_captures([cam_1, cam_2])
    while True:
        ret_1, frame_1 = cam_1.get_frame()
        ret_2, frame_2 = cam_2.get_frame()

        ret_1, y_median_1, x_median_1 = detector.detect(frame_1)
        ret_2, y_median_2, x_median_2 = detector.detect(frame_2)

        if ret_1:
            cv.circle(frame_1, (x_median_1, y_median_1), 8, (0, 0, 255), -1)
            ray_1 = cam_1.point_to_ray((x_median_1, y_median_1))
        if ret_2:
            cv.circle(frame_2, (x_median_2, y_median_2), 8, (0, 0, 255), -1)
            ray_2 = cam_2.point_to_ray((x_median_2, y_median_2))

        if ret_1 and ret_2:
            predicted_point = lsl.predict([ray_1, ray_2], [1, 1])
            # print(f"Predicted ball position: {predicted_point}")
            detected_frames += 1
        else:
            detected_frames -= 1
        detected_frames = min(max(0, detected_frames), detected_frames_cap)

        if detected_frames > detected_frame_threshold:
            if not detected:
                detected = True
            t = time() - detection_start_time
            x_rpf.add_point(t, predicted_point[0])
            y_rpf.add_point(t, predicted_point[1])
            z_rpf.add_point(t, predicted_point[2])
        else:
            if detected:
                detected = False
                print(f"Detection that started at {detection_start_time}")
                print(f"{x_rpf.get_coef().round(3) = }")
                print(f"{y_rpf.get_coef().round(3) = }")
                print(f"{z_rpf.get_coef().round(3) = }")
            x_rpf.reset()
            y_rpf.reset()
            z_rpf.reset()
            detection_start_time = time()

        cv.imshow("Left view", frame_1)
        cv.imshow("Right view", frame_2)
        plotter.draw_point(predicted_point)

        if cv.waitKey(1) == ord("q"):
            break

    release_captures([cam_1, cam_2])
    cv.destroyAllWindows()


if __name__ == "__main__":
    # single_camera_test()
    double_camera_test()