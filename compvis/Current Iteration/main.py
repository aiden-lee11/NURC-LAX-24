import cv2
from time import time
import numpy as np
from scipy.spatial.transform import Rotation as R
from camera import camera_instantiator
from timers import timers
from triangulation import LSLocalizer
from predict import RecursivePolynomialFit
from visualization import PointInSpace


# send updated position in the plane of the robot, coordinate frame using, 

def calculate_points(lsl, rays, calculated_pts):
    ray_vals = np.array(list(rays.values()))
    calculated_pt = lsl.predict(ray_vals)
    calculated_pts.append(calculated_pt)
    print(f"Calculated point: {calculated_pt}")
    return calculated_pt


def main_loop(cameras, lsl, static):
    calculated_pts = []
     
    lim_x = [-1, 1]
    lim_y = [0, 2]
    lim_z = [-0.5, 0.5]
    plotter = PointInSpace(lim_x, lim_y, lim_z)
    detection_start_time = time()   
    t = time() - detection_start_time

    detected_frames = 0
    detected_frames_cap = 30
    detected_frame_threshold = 5
    detected = False
    x_rpf = RecursivePolynomialFit(2)
    y_rpf = RecursivePolynomialFit(2)
    z_rpf = RecursivePolynomialFit(2)
    
    

    while True:
        with timers.timers["Main Loop"]:
            if cv2.waitKey(1) == ord("q"):
                for camera in cameras.values():
                    camera.release_camera()
                break

            rays = {
                camera: result
                for camera in cameras.values()
                if (result := camera.run()) is not None
            }


            if rays:
                calculated_point = calculate_points(lsl, rays, calculated_pts)
                plotter.draw_point(calculated_point)

                # print(f"Predicted ball position: {predicted_point}")

                detected_frames += 1
            else:
                detected_frames -= 1

            detected_frames = min(max(0, detected_frames), detected_frames_cap)

            if detected_frames > detected_frame_threshold:
                detected = True
                t = time() - detection_start_time
                x_rpf.add_point(t, calculated_point[0])
                y_rpf.add_point(t, calculated_point[1])
                z_rpf.add_point(t, calculated_point[2])

                intersection_time = y_rpf.solve(0).round(3)[1] # inter_time gets [negative positive] only need positive
                #print(f"intersection time = {intersection_time}")
                x_predicted = x_rpf.plug_in(intersection_time)
                z_predicted = z_rpf.plug_in(intersection_time)

                #print(f"x_coord prediction = {x_predicted}")
                #print(f"z_coord prediction = {z_predicted}")

                continue


            elif detected:
                detected = False
                print(f"Detection that started at {detection_start_time}")
                print(f"{x_rpf.get_coef().round(3) = }")
                print(f"{y_rpf.get_coef().round(3) = }")
                print(f"{z_rpf.get_coef().round(3) = }")
                x_rpf.reset()
                y_rpf.reset()
                z_rpf.reset()
            detection_start_time = time()


        timers.record_time("Main Loop")


def main(camera_transforms, cam_ids=None):
    static_hsv = input('Use static hsv values? (y/n): ').lower().strip() == 'y'
    cameras = camera_instantiator(cam_ids, static_hsv)
    print("Press q to release cameras and exit.\n")
    lam = 0.98
    lsl = LSLocalizer(camera_transforms)
    main_loop(cameras, lsl, static_hsv)

    cv2.destroyAllWindows()
    timers.display_averages()


if __name__ == "__main__":
    # first camera at origin
    T_cam1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # second camera rotated pi/2 about Z at (1, 1, 0)
    T_cam2 = np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


    #first camera at origin
    # T_cam1 = np.array(
    #     [
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 0],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 1],
    #     ]
    # )
    # # second camera unrotated about Z at (0, 1, 0)
    # T_cam2 = np.array(
    #     [
    #         [1, 0, 0, 0],
    #         [0, 1, 0, 1],
    #         [0, 0, 1, 0],
    #         [0, 0, 0, 1],
    #     ]
    # )
    camera_transforms = [T_cam1, T_cam2]

    main(camera_transforms)
