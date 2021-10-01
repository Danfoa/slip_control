#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/10/21
# @Author  : Daniel Ordonez 
# @email   : daniels.ordonez@gmail.com
from math import pi as PI
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from slip_control.slip.slip_trajectory import SlipTrajectory
from slip_control.slip.slip_model import SlipModel, X, X_DOT, X_DDOT, Z, Z_DOT, Z_DDOT
from slip_control.controllers.target_to_states_generator import CycleStateGenerator, ForwardSpeedStateGenerator
from slip_control.controllers.diff_flat_slip_controller import SlipDiffFlatController
from slip_control.utils import plot_utils

cmap = plt.cm.get_cmap('gist_heat')

if __name__ == "__main__":

    # Instantiate SLIP model
    m = 80  # [kg]
    r0 = 1.0  # [m]
    n_legs = 1
    k_rel = 10.7
    slip = SlipModel(mass=m, leg_length=r0, k_rel=k_rel * n_legs)

    g = SlipModel.g

    # Error deviation weights during the stance trajectory
    traj_weights = np.array([1., 1., 1., 1., 1., 1.])
    traj_weights /= np.linalg.norm(traj_weights)
    # Error deviation weights of target take-off states
    take_off_state_error_weights = np.array([0.0, 1.0, 0., 1.0, 1.0, 0.])
    take_off_state_error_weights /= np.linalg.norm(take_off_state_error_weights)


    n_cycles = 5 # Generate a trajectory of 5 cycles
    max_theta_dot = 4*PI # [rad/s] max angular leg velocity during flight
    # Define a forward velocity
    forward_speed = 4 * slip.r0  # [m/s]
    # Define a desired gait duty cycle (time of stance / time of cycle) in [0.2, 1.0]
    duty_cycle = 0.8

    z_init = slip.r0
    # Set an initial state (assumed to be a flight phase state)  [x, x', x'', z, z', z'']
    init_to_state = np.array([0.0, forward_speed, 0.0, z_init, 0.0, -g])
    # Set a desired take off state defining the forward and vertical velocity desired
    to_des_state = init_to_state

    # Configure Differentially flat controller
    slip_controller = SlipDiffFlatController(slip_model=slip,
                                             traj_weights=traj_weights,
                                             max_flight_theta_dot=max_theta_dot,
                                             debug=False)
    to_state_generator = ForwardSpeedStateGenerator(slip_model=slip, target_state_weights=take_off_state_error_weights,
                                                    desired_forward_speed=forward_speed,
                                                    desired_duty_cycle=duty_cycle)
    slip_controller.target_to_state_generator = to_state_generator

    # Generate SLIP trajectory tree without future cycle planning
    tree = slip_controller.generate_slip_trajectory_tree(desired_gait_cycles=n_cycles,
                                                         initial_state=init_to_state,
                                                         max_samples_per_cycle=30,
                                                         angle_epsilon=np.deg2rad(.02),
                                                         look_ahead_cycles=0)
    slip_traj_no_future = tree.get_optimal_trajectory()
    plot_utils.plot_slip_trajectory(slip_traj_no_future, plot_passive=True, plot_td_angles=True,
                                          title="Without future cycle planning",
                                          color=(23/255., 0/255., 194/255.))
    plt.show()

    # Generate SLIP trajectory tree with future cycle planning
    tree = slip_controller.generate_slip_trajectory_tree(desired_gait_cycles=n_cycles,
                                                         initial_state=init_to_state,
                                                         max_samples_per_cycle=30,
                                                         angle_epsilon=np.deg2rad(.02),
                                                         look_ahead_cycles=1)
    slip_traj = tree.get_optimal_trajectory()
    plot_utils.plot_slip_trajectory(slip_traj, plot_passive=True, plot_td_angles=True,
                                          title="With future cycle planing",
                                          color=(23 / 255., 154 / 255., 194 / 255.))
    plt.show()

    # Plot controlled trajectory tree
    print("This takes a while... should optimize soon")
    tree.plot()
    plt.show()

    # Compare first two cycles.
    short_traj = SlipTrajectory(slip, slip_gait_cycles=slip_traj.gait_cycles[:2])
    short_traj_no_future = SlipTrajectory(slip, slip_gait_cycles=slip_traj_no_future.gait_cycles[:2])
    axs = plot_utils.plot_slip_trajectory(short_traj, plot_passive=True, plot_td_angles=True,
                                          color=(23 / 255., 154 / 255., 194 / 255.))
    axs = plot_utils.plot_slip_trajectory(short_traj_no_future, plot_passive=True, plot_td_angles=True, plt_axs=axs,
                                          color=(23/255., 0/255., 194/255.))
    plt.show()

    # Plot limit cycles of controlled trajectory
    phase_axs = plot_utils.plot_limit_cycles(slip_traj)
    plt.show()