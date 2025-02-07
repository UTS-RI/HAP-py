# HAP-py
HAP python library. Computes one or multiple epsilon-Gromov Hausdorff Approximations (e-GHAs or HAs for short) by running an optimisation routine described in the paper "Multi-query Robotic Manipulator Task Sequencing with Gromov-Hausdorff Approximations". A HA is a mapping between two metric spaces such that when a geodesic (shortest path) in one space is mapped through to the other its path length is approximately preserved by some bounded amount, epsilon. In the context of a robot, HAs are useful for identifying points that are close to one another in both task and configuration space. This library takes as input a discretised task space and environment and outputs one or multiple covering HAs. 

Conceptually, HAs can be seen as a subspace decomposition of the task space where every point within a subspace is assigned a one-to-one mapping (bijection) to the configuration space such that any path through this subspace satisfies our bounded path length condition. Subspaces are represented as undirected subgraphs based on the task space. This library uses networkx data structure for storing these subspace graphs.

An example usage of HAs is for reducing the search space of a motion planner and ensuring no sudden reconfiguration of the robot. Another use case is task sequencing. Note that HAs are only valid for obstacles given at computation time. If these change then HAs need to be recomputed or paths need to be adapted to compensate for this using, for example, a trajectory optimiser.

# Setup
```
cd HAP-py
pip install -e .
```
Note that if you would like to run the example.py script you need to install ur_ikfast from source: https://github.com/cambel/ur_ikfast

# Usage
See example.py for how to use

# Citation

Multi-query Robotic Manipulator Task Sequencing with Gromov-Hausdorff Approximations (conditionally accepted into TRO): https://arxiv.org/abs/2209.04800
