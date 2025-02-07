# Copyright 2025 Fouad Sukkar
#
# This code is licensed under MIT license (see LICENSE.txt for details)

""" Hausdorff Approximation Generator

Given a task space (6D poses) and corresponding valid, collision free IK solutions, finds one or multiple covering subspaces where each pose
within a subspace maps to a unique arm configuration.
The resulting mapping is an epsilon-Gromov Hausdorff Approximation (eGHA) where the difference in distance between a path in the task space
and its mapped path in the configuration space are bounded by a chosen epsilon.
This eGHA is approximated by a roadmap consisting of nodes (poses and mapped configurations) and edges between them.
"""

import numpy as np
import heapq
from scipy.spatial import KDTree
from scipy.spatial.transform import Rotation as R
import networkx as nx
from networkx.readwrite import json_graph
import pickle
import yaml
import timeit
import os


class HAP():
    """
    Class used to compute HAs. Subspaces and maps are represented by list of subgraphs based on the given task space.

    """

    def __init__(self, collision_detector, check_ray_collision, task_space, kinematics, params_file):
        """

        @param collision_detector collision detector handle which returns true if config is in collision
        @param check_ray_collision for culling edges that are in collision with obstacles (prior to subspace optimisation)
        @param task_space list of poses approximating the task/ operational space from which the roadmap will be constructed over
        @param kinematics robot kinematics solver, should provide methods "forward" and "inverse" with similar arguments to ur_ikfast's
        @param params_file yaml file with HAP parameters
        """
        with open(params_file, 'r') as file:
            params = yaml.safe_load(file)
        try: 
            self.ee_offset = params['ee_offset']
        except:
            self.ee_offset = np.eye(4)
        self.collision_interp_res = params['collision_interp_res']
        self.epsilon = params['epsilon']
        self.max_grid_edge_dist = params['max_grid_edge_dist']
        self.max_num_subspaces = params['max_num_subspaces']
        self.num_root_nodes = params['num_root_nodes']
        self.c_max = params['c_max']
        self.zeta = params['zeta']
        self.kinematics = kinematics
        self.collision_detector = collision_detector
        self.check_ray_collision = check_ray_collision
        self.global_min_unreachable = np.inf  # minimum objective cost found by modified Dijkstra routine
        self.prm_tree = None
        self.task_space = task_space
        self.num_subspaces = 0
        self.valid_poses = []  # task space poses with alteast one valid IK solution
        self.valid_poses_inds = []  # task space pose indices of the valid poses
        self.HAs = []  # list of HA maps
        self.avg_config = []

    def decompose_task_space(self):
        print("computing IK solutions for task space poses...")
        self.valid_poses_inds, self.valid_poses, poses_solutions = self.get_ik_solutions(self.task_space)
        print("number of valid IK solutions: " + str(len(self.valid_poses)))
        T_open_inds = range(len(self.valid_poses))     # start_nodes
        # unmapped_tasks_inds = range(len(self.valid_poses))   # raw_nodes_tracker_inds
        mapped_tasks_counter = np.zeros(len(self.valid_poses)).tolist()   # self.raw_nodes_tracker
        solutions_kdtrees = [KDTree(solutions) for solutions in poses_solutions]
        positions = np.array(self.valid_poses)[:,:3,3]
        positions_kdtrees = KDTree(positions)
        poses_edges = self.gen_edges(positions, positions_kdtrees)

        while self.num_subspaces < self.max_num_subspaces:
            print("Iteration " + str(self.num_subspaces) + " of task space decomposition")
            J_min = np.inf
            min_mapped_configs = {}
            min_path_costs = []
            min_edges_reachable = []
            min_edges_reachable_costs = []
            if self.num_root_nodes == -1:
                T_root_inds = T_open_inds[:]
            else:
                if len(T_open_inds):
                    T_root_inds = np.random.choice(T_open_inds, size=min(self.num_root_nodes, len(T_open_inds)), replace=False)
                else:
                    rospy.loginfo("No more unmapped nodes remaining (full valid task space coverage achieved), exiting...")
                    sys.exit()
            for root_node_ind in T_root_inds:
                start_time_root_node = timeit.default_timer()
                try:
                    mapped_configs, path_costs, edges_reachable, edges_reachable_costs, J_root_min = self.generate_map(root_node_ind, self.valid_poses, poses_solutions, solutions_kdtrees, positions_kdtrees, poses_edges)
                except Exception as ex:
                    J_root_min = np.inf
                    mapped_configs = {}
                    path_costs = []
                    edges_reachable = []
                    edges_reachable_costs = []
                    print(ex)
                print("map generation time (s) for root node " + str(root_node_ind) + ": " + str(timeit.default_timer() - start_time_root_node))
                if J_root_min < J_min:
                    print("new minimum cost: " + str(J_root_min))
                    J_min = J_root_min
                    min_mapped_configs = mapped_configs.copy()
                    min_path_costs = path_costs[:]
                    min_edges_reachable = edges_reachable[:]
                    min_edges_reachable_costs = edges_reachable_costs[:]
            
            print("number of nodes in subspace " + str(self.num_subspaces) + ": " + str(len(min_mapped_configs.keys())))

            T_open_inds = [ind for ind in T_open_inds if ind not in min_mapped_configs.keys()]
            print("number of unmapped nodes left (T_open): " + str(len(T_open_inds)))

            if self.num_subspaces == 0:
                self.avg_config = np.mean(list(min_mapped_configs.values()), axis=0)

            if len(min_mapped_configs):
                roadmap = self.gen_nx_graph(min_mapped_configs, min_path_costs, min_edges_reachable, min_edges_reachable_costs)
                self.HAs.append(roadmap)
            self.num_subspaces += 1

    def save_HAs(self, directory='databases/'):
        HAs_dicts = [json_graph.node_link_data(subspace_graph) for subspace_graph in self.HAs]
        file_dir = os.path.dirname(directory + 'HAs.pkl')
        if file_dir != '' and not os.path.isdir(file_dir):
            raise FileNotFoundError("directory for saving HAs does not exist...")

        with open(directory + 'HAs.pkl', 'wb') as file:
            pickle.dump(HAs_dicts, file)
        print('saved roadmap to: ' + str(directory + 'HAs.pkl'))

        R_valid_poses = R.from_matrix(np.array(self.valid_poses)[:, :3, :3])
        positions = np.array(self.valid_poses)[:, :3, 3]
        quats = R_valid_poses.as_quat()
        pos_quat = np.concatenate((positions, quats), axis=1)  # x, y, z, qx, qy, qz, qw
        np.savetxt(directory + '/valid_poses.txt', pos_quat, delimiter=',', header="x, y, z, qx, qy, qz, qw")

    @staticmethod
    def load_HAs(directory='databases/', create_using=nx.Graph):    
        with open(directory + 'HAs.pkl', 'rb') as file:
            HAs_dicts = pickle.load(file)
        HAs = [json_graph.node_link_graph(subspace_graph) for subspace_graph in HAs_dicts]
        valid_poses = np.genfromtxt(directory + 'valid_poses.txt', delimiter=',')
        return valid_poses, HAs

    def gen_nx_graph(self, mapped_configs, path_costs, edges_reachable, edges_reachable_costs):
        g = nx.Graph()
        g.add_nodes_from(mapped_configs.keys())
        for n in g.nodes:
            g.nodes[n]['config'] = mapped_configs[n]
        for i, edge in enumerate(edges_reachable):
            g.add_edge(edge[0], edge[1], weight=edges_reachable_costs[i])
        return g

    def add_node_to_prm(self, g, config, n_neighbours=5):
        node_num = max(g.nodes) + 1 
        g.add_node(node_num, config=np.array(config))

        # get 5 closest nodes to start and end and connect them
        edge_weights, node_idx = self.prm_tree.query(config, k=n_neighbours)
        for i in range(n_neighbours):
            g.add_edge(node_num, self.configs_to_node_mapping[node_idx[i]], weight=edge_weights[i])
        return node_num

    def plan(self, g, start_config, goal_config):    # A* heuristic
        if self.prm_tree is None:
            print('constructing KDTree from configs since first time planning...')
            configs = np.array([g.nodes[i]['config'] for i in g.nodes])
            self.configs_to_node_mapping = [i for i in g.nodes]
            self.prm_tree = KDTree(configs)
        g2 = g.copy()
        prm_start_index = self.add_node_to_prm(g2, start_config)
        prm_end_index = self.add_node_to_prm(g2, goal_config)

        def heuristic(a, b):
            return np.linalg.norm(g2.nodes[a]['config'] - g2.nodes[b]['config'])
        try:
            path = nx.astar_path(g2, prm_start_index, prm_end_index, weight='weight', heuristic=heuristic)
        except Exception as ex:
            print(ex)
            print('No path found')
            return None
        return [g2.nodes[ind]['config'] for ind in path]

    def collision_check_edge(self, config1, config2):
        dist = np.linalg.norm(config1 - config2)
        lin_configs = np.linspace(config1, config2, int(np.round(dist / self.collision_interp_res)))
        for config in lin_configs:
            if self.collision_detector(config):
                return True
        return False


    def generate_map(self, root_node_ind, valid_poses, poses_solutions, solutions_kdtrees, positions_kdtrees, poses_edges):
        J_config_min = np.inf
        max_dist_avg_config_flag = True
        for source_solution in poses_solutions[root_node_ind]:
            if len(self.avg_config):
                dist_avg_config = np.linalg.norm(self.avg_config - source_solution)
                if dist_avg_config > self.zeta:
                    continue
                max_dist_avg_config_flag = False
            else:
                max_dist_avg_config_flag = False
            path_costs = [self.c_max for pose in valid_poses]
            poses_solutions_temp = poses_solutions[:]  # risky because if modify anything lower than 1st index will modify original pose_solutions
            path_costs[root_node_ind] = 0.0
            edges_reachable = []
            edges_reachable_costs = []

            parents = {}
            mapped_configs = {}
            mapped_configs[root_node_ind] = source_solution
            Qlist_maintainer = [0 for x in valid_poses]  # 0 means pose is not on the heapq, 1 means it is
            parents[(root_node_ind)] = (root_node_ind, source_solution)
            Qlist = [[0.0, root_node_ind, source_solution]]  # open list
            heapq.heapify(Qlist)

            # algorithm from here runs similar to standard Dijkstra except during the node expansion
            while len(Qlist) > 0:
                u = heapq.heappop(Qlist)
                current_joints = u[2]
                current_ind = u[1]
                Qlist_maintainer[current_ind] = 0

                # expand node
                neighbour_inds = poses_edges[current_ind]
                no_solutions = True
                # iterate over neighbours
                for neighbour_ind in neighbour_inds:
                    neighbour_solutions = poses_solutions_temp[neighbour_ind]
                    if neighbour_ind in mapped_configs:
                        min_cost = max([abs(x - y) for x, y in zip(current_joints.tolist(), mapped_configs[neighbour_ind].tolist())])
                        if min_cost < self.epsilon and not self.collision_check_edge(current_joints, mapped_configs[neighbour_ind]):
                            min_ind = 0  # since there should only be one solution now in neighbour_solutions
                            edges_reachable.append([current_ind, neighbour_ind])
                            edges_reachable_costs.append(min_cost)
                        else:
                            min_cost = np.inf
                            min_ind = len(neighbour_solutions)
                    else:
                        # retrieve IK solutions of neighbour
                        tree = solutions_kdtrees[neighbour_ind]
                        valid_inds = tree.query_ball_point(current_joints, p=np.inf, r=self.epsilon) 
                        min_ind = len(neighbour_solutions)
                        min_cost = np.inf
                        for valid_ind in valid_inds:
                            # TODO: need to change eventually to consider edge distance in task space, currently assumes all edges are roughly the same distance
                            # (value in paper is different because here the max edge cost is added to epsilon)
                            cost = max([abs(x - y) for x, y in zip(current_joints.tolist(), neighbour_solutions[valid_ind].tolist())])
                            if cost < min_cost:
                                min_ind = valid_ind
                                min_cost = cost

                    # if a candidate mapping was found that did not violate the e-GHA constraint
                    # assign node mapping and update path costs if a shorter path was found
                    if min_ind != len(neighbour_solutions):
                        no_solutions = False
                        min_neighbour_config = neighbour_solutions[min_ind]
                        path_cost = path_costs[current_ind] + min_cost

                        if path_costs[neighbour_ind] > path_cost:
                            poses_solutions_temp[neighbour_ind] = [neighbour_solutions[min_ind]]
                            path_costs[neighbour_ind] = path_cost
                            mapped_configs[neighbour_ind] = min_neighbour_config
                            parents[neighbour_ind] = (current_ind, current_joints)

                            if Qlist_maintainer[neighbour_ind] == 0:
                                heapq.heappush(Qlist, [path_cost, neighbour_ind, min_neighbour_config])
                                Qlist_maintainer[neighbour_ind] = 1

            J_config = np.sum(path_costs)

            if J_config < J_config_min:
                J_config_min = J_config
                min_mapped_configs = mapped_configs.copy()
                min_path_costs = path_costs[:]
                min_edges_reachable = edges_reachable[:]
                min_edges_reachable_costs = edges_reachable_costs[:]


            if len(mapped_configs) == 1:
                print("unable to connect any poses, try increasing the density of task space or increase max_grid_edge_dist")


        if max_dist_avg_config_flag:
            raise Exception("could not find an IK solution within the zeta threshold for node " + str(root_node_ind) + " as root node")

        # print("unmapped pose indices due to eGHA violation:")
        # for ind, solutions in enumerate(poses_solutions_temp):
        #     if ind not in mapped_configs:
        #         print(ind)

        return mapped_configs, path_costs, edges_reachable, edges_reachable_costs, J_config_min



    def gen_edges(self, positions, positions_kdtrees):
        """ Generates a graph over the given list of poses based on set max_grid_edge_dist.

        Generates a graph over the given list of poses based on set max_grid_edge_dist. In order for an edge to be created between
        two poses, their distance (position only) should be within the set max_grid_edge_dist and there should also
        be valid path in which the arm can travel along this edge.

        @param poses list of poses
        @return edges with valid connections
        """
        edges = positions_kdtrees.sparse_distance_matrix(positions_kdtrees, self.max_grid_edge_dist)

        poses_valid_edges = []
        for current_index in range(len(positions)):
            # iterate over neighbouring edges
            keys = edges[current_index, :].keys()
            indices = [x[1] for x in keys]
            valid_edges = []
            for neighbour_index in indices:
                if not self.check_ray_collision(positions[current_index], positions[neighbour_index]):  # TODO: implement this. simple ray collision or consider full geometry of arm?
                    valid_edges.append(neighbour_index)
            poses_valid_edges.append(valid_edges)
        return poses_valid_edges

    def get_ik_solutions(self, poses):
        """ Returns all valid and collision-free ik solutions for each pose.

        Returns all valid and collision-free ik solutions for each pose. Returned solutions do not contain empty solutions.

        @param poses list of poses to compute ik solutions for
        @return list of valid and collision-free ik solutions for each pose and list of corresponding poses
        """
        valid_poses = []
        valid_poses_inds = []
        poses_solutions = []
        for ind, pose in enumerate(poses):
            offset_pose = np.dot(pose, np.linalg.pinv(self.ee_offset))
            solutions = self.kinematics.inverse(offset_pose[:3,:], True)  # pose is [x, y, z, qw, qx, qy, qz] or 3x4 transform matrix
            if len(solutions):
                valid_solutions = []
                for solution in solutions:
                    if not self.collision_detector(solution):
                        valid_solutions.append(solution)
                if len(valid_solutions):
                    poses_solutions.append(valid_solutions)
                    valid_poses.append(pose)
                    valid_poses_inds.append(ind)
        return valid_poses_inds, valid_poses, poses_solutions
