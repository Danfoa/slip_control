# Spring-loaded Inverted Pendulum (SLIP) Control

This repository intends to provide a base for the implementation, visualization, and *open-access* sharing of
SLIP control for research and education. The SLIP model is a bio-inspired low-order template model for dynamics of
(high-speed) legged locomotion.  <cite>[[1]][1]</cite> It has been proven valuable in modeling the sagittal dynamics of the Center of Mass (CoM)
of animals at high-speed locomotion, and extensions of this model were to model 3D bipedal walking and running.

![slip_repo_into](https://user-images.githubusercontent.com/8356912/135597417-ea86cddf-d9bd-4dc6-8f34-6a32abb94521.png)

https://user-images.githubusercontent.com/8356912/135605031-1d53dafc-bf50-4a66-b331-90fb9c892caf.mp4

*Motivation: There is a lot of research and educational content based on the SLIP model but no shared code or initiative
that allows playing with the model control without implementing everything from scratch. The ideal scenario is for this repo to become a mechanism for people to contribute different approaches to SLIP control, visualization, debugging,
and educational tools.*

### Main Features
The repository offers python class objects with several utility functions to create SLIP models and plan passive or controlled trajectories, plus visualization tools for these trajectories. The main classes are:

- `SlipModel`: a python class representation of a SLIP model. The class enables the generation of passive
  trajectories by integration of the hybrid-dynamics differential equations governing SLIP, and some additional utility
  functions for visualization and state handling.

- `SlipGaitCycle` and `SlipGaitCycleCtrl`: python classes representing a full gait cycle (flight and stance phases) of
  a SLIP model. The class provides an intuitive interface to storing the cartesian and polar trajectories of the SLIP
  gait cycle model, along with accessors to full-cycle cartesian trajectories, foot contact positions, touch-down/take-off leg angles, and more. The controlled version is intended to store both the passive and controlled trajectories of the model.
- `SlipTrajectory`: a class containing a sequence of consecutive `SlipGaitCycle`s, ensuring continuity in the trajectory. This class helps in saving and restoring a trajectory, plotting, and more.

- `SlipTrajectoryTree`: a tree data structure intended to be used to plan multi-cycle trajectories.
  It offers the capability to store multiple possible trajectories emerging from different future touch down angles,
  selection of the most optimal and some plotting utilities.

- Plotting: both for control development, debugging and especially for education, visual aid of the model dynamics and response. In utils, you will find plotting and (soon) animating utility functions to generate rich visualizations of
  `SlipTrajectory`s and `SlipTrajectoryTree`s.

The structure and functions of the classes are based on what I found useful when developing a multi-cycle
trajectory planner for SLIP. Any modifications and enhancements are more than welcome. Submit a PR! :).

### Controllers

In the `slip_control/controllers` you will find single-cycle and multi-cycle trajectory control controllers. There
is only one controller so far, but I encourage you to contribute many:

- [x] Differentially Flat optimal control of SLIP: A controller of an extended version of the SLIP model using hip torque and leg force actuation. Details in [[3]] (Chapter 4.1). This controller is an adapted version of [[2]].

### Instalation

This repo is still in early development therefore, there is no release, but installation is easy through pip via:

```buildoutcfg
# Clone repo 
git clone https://github.com/Danfoa/slip_control.git
# Move to repo folder 
cd slip_control 
# Use pip to install repo and dependencies
pip install -e .
```

#### Examples

A preliminary example of the visualization tools and the Differentially Flat Controller can be run by 

```buildoutcfg
cd <repo>
# Run example
python examples/diff_flat_control_and_visualization.py 
```


### References and recommended literature

By far, the best summary of the SLIP model and its control is [[1]]. For the model, biomechanical foundations [[4]] offers a
a great introduction to the value of SLIP as a generic locomotion model.

- [[1]] Geyer, H., et al. "Gait based on the spring-loaded inverted pendulum." Humanoid Robotics: A Reference. Springer Netherlands, 2018. 1-25.
- [[2]] Chen, Hua, Patrick M. Wensing, and Wei Zhang. "Optimal control of a differentially flat two-dimensional spring-loaded inverted pendulum model." IEEE Robotics and Automation Letters 5.2 (2019): 307-314.
- [[3]] Ordoñez Apraez, Daniel Felipe. Learning to run naturally: guiding policies with the Spring-Loaded Inverted Pendulum. MS thesis. Universitat Politècnica de Catalunya, 2021.
- [[4]] Full, Robert J., Claire T. Farley, and Jack M. Winters. "Musculoskeletal dynamics in rhythmic systems: a comparative approach to legged locomotion." Biomechanics and neural control of posture and movement. Springer, New York, NY, 2000. 192-205.


[1]: https://link.springer.com/referenceworkentry/10.1007%2F978-94-007-7194-9_43-1#:~:text=The%20spring%2Dloaded%20inverted%20pendulum%20(SLIP)%20describes%20gait%20with,control%20of%20compliant%20legged%20locomotion. "Hello"
[2]: https://ieeexplore.ieee.org/abstract/document/8917684?casa_token=u34S9aFjkVAAAAAA:JqTD1t2tg4w_Xojfzc18fPggjZGO3mztxcMUzrY-pzuCeqwhvxjL5Dz7lHbFgkgOwUVzu6YZsEE
[3]: https://upcommons.upc.edu/handle/2117/348196
[4]: https://link.springer.com/chapter/10.1007/978-1-4612-2104-3_13
