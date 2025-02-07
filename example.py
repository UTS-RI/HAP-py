# Copyright 2025 Fouad Sukkar
#
# This code is licensed under MIT license (see LICENSE.txt for details)


from HAP_py.HAP import HAP
import networkx as nx
from task_space_example import gen_task_space
from ur_ikfast import ur_kinematics
import numpy as np
import os

DATABASE_DIR = 'databases/test/'
PARAMS_FILE = 'config/hap_params.yaml'

def collision_detector(config):
    return False

def check_ray_collision(pos1, pos2):
    return False

if __name__ == "__main__":
    # eGHA construction - subspaces and maps are represented by list of subgraphs based on the given task space
    task_space = gen_task_space()
    kinematics = ur_kinematics.URKinematics('ur5')
    hap  = HAP(collision_detector, check_ray_collision, task_space, kinematics, PARAMS_FILE)
    # only need to run this once for a particular task space/ environment
    if not os.path.isdir(DATABASE_DIR):
        os.makedirs(DATABASE_DIR, exist_ok=True)  # only works python >= 3.2
    hap.decompose_task_space()
    hap.save_HAs(DATABASE_DIR)

    # iterate through subspaces and inspect poses and configs
    valid_poses, HAs = hap.load_HAs(DATABASE_DIR)
    for subspace_graph in HAs:
        subspace_poses = np.array(valid_poses)[list(subspace_graph.nodes)]
        subspace_configs = [subspace_graph.nodes[ind]['config'] for ind in list(subspace_graph.nodes)]

    # example of how to plan using HAs
    subspace_configs = [HAs[0].nodes[ind]['config'] for ind in list(HAs[0].nodes)]
    idx = np.random.randint(len(subspace_configs), size=2)
    start_config = subspace_configs[idx[0]]
    goal_config = subspace_configs[idx[1]]
    trajectory = hap.plan(HAs[0], start_config, goal_config)
    print(trajectory)
