from camera import camera_instantiator
from triangulation import LSLocalizer
import numpy as np

def ray_test():
    T_cam1 = np.array(
        [
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    # second camera non rotated at (1,0,0)
    T_cam2 = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )

    camera_transforms = [T_cam1, T_cam2]

    ray_1 = np.array([.5, 1, 0])
    ray_2 = np.array([-.5, 1, 0])

    ray_vals = [ray_1, ray_2]
    lsl = LSLocalizer(camera_transforms)

    print(lsl.predict(ray_vals))

def camera_ball_test():
    cameras = camera_instantiator()
    camer
