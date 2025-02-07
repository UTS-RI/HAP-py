# Copyright 2025 Fouad Sukkar
#
# This code is licensed under MIT license (see LICENSE.txt for details)


import numpy as np

def gen_task_space():
    """ Generates discrete set of poses to form the task space.

    Generates discrete set of poses, manually defined here as uniform grid facing into the world -z direction with 45 deg offsets.

    @return list of poses
    """
    poses = []
    ats = [
        [0.0, 0.0, -1.0],
        [0.0, -1.0, -1.0],
        [-1.0, 0.0, -1.0],
        [0.0, 1.0, -1.0],
        [1.0, 0.0, -1.0],
    ]
    up_vector = [-1.0, 0.0, 0.0]

    step = 0.1
    pos_x_list = np.arange(0.25, 0.85 + step, step)
    pos_y_list = np.arange(-0.45, 0.45 + step, step)
    pos_z_list = np.arange(0.15, 0.45 + step, step)

    for pos_x in pos_x_list:
        for pos_y in pos_y_list:
            for pos_z in pos_z_list:
                for at_offset in ats:
                    eye = [pos_x, pos_y, pos_z]
                    at = [pos_x + 0.001, pos_y - 0.001, pos_z]  # 0.001 is because IKFast solution is singular for poses pointing directly in z axis
                    at = [x + y for x, y in zip (at, at_offset)]
                    T = transform_lookat(at, eye, up_vector)
                    poses.append(T)
    return poses

def transform_lookat(at, eye, up):
    """ Copied from OpenRAVE's transformLookat function in "geometry.h".

    Returns an end effector transform matrix that looks along a ray with a desired up vector (corresponding to y axis of the end effector).
    If up vector is parallel to ray, tries to use +y or +x direction instead.
    If ray length is zero, chooses ray to be +z direction by default.

    @param at the point space to look at, the camera will rotation and zoom around this point
    @param eye the position of the camera in space
    @param up desired end effector y axis direction
    @return end effector transform matrix
    """
    vdir = np.array(at) - eye
    if np.linalg.norm(vdir) > 1e-6:
        vdir *= 1 / np.linalg.norm(vdir)
    else:
        vdir = [0.0, 0.0, 1.0]

    vup = np.array(up) - vdir * np.dot(up, vdir)
    if np.linalg.norm(vup) < 1e-8:
        vup = [0.0, 1.0, 0.0]
        vup -= vdir * np.dot(vdir, vup)
        if np.linalg.norm(vup) < 1e-8:
            vup = [1.0, 0.0, 0.0]
            vup -= vdir * np.dot(vdir, vup)

    vup *= 1 / np.linalg.norm(vup)
    right = np.cross(vup, vdir)

    rot_mat = np.transpose([right, vup, vdir])
    T = [list(rot_mat[0]) + [eye[0]],
         list(rot_mat[1]) + [eye[1]],
         list(rot_mat[2]) + [eye[2]],
         [0, 0, 0, 1]]
    return np.array(T)