# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.13 |Anaconda, Inc.| (default, Feb 23 2021, 21:15:04) 
# [GCC 7.3.0]
# Embedded file name: /home/dordonez/Projects/slip_control/slip_control/controllers/diff_flat_slip_controller.py
# Compiled at: 2021-09-30 17:51:21
# Size of source mod 2**32: 34875 bytes
import sys
import time
from operator import xor
from typing import Union, Optional, Dict

import cvxpy as cp
import numpy as np
import scipy
from scipy.linalg import block_diag
from tqdm import tqdm

from .target_to_states_generator import SlipTargetStateGenerator
from ..slip.slip_gait_cycle import SlipGaitCycle, SlipGaitCycleCtrl
from ..slip.slip_model import SlipModel, MAX_TD_ANGLE, MIN_TD_ANGLE
from ..slip.slip_model import X, X_DOT, X_DDOT, Z, Z_DOT, Z_DDOT, R
from ..slip.slip_trajectory_tree import TrajectoryTreeNode

g = SlipModel.g


class SlipDiffFlatController:

    def __init__(self, slip_model: SlipModel, traj_weights,
                 target_to_state_generator: Optional[SlipTargetStateGenerator] = None, poly_degree=6, qp_grid_points=11,
                 max_flight_theta_dot=np.inf, debug=False):
        if not 32 > poly_degree >= 6:
            raise AssertionError('Minimum polynomial degree is 6 max is 32, 6 is enough!')
        else:
            assert traj_weights.shape == (6,), 'Invalid costs shapes [%s], should (6,)' % str(traj_weights.shape)
            if target_to_state_generator is None:
                target_to_state_generator = SlipTargetStateGenerator(slip_model, np.ones(6))
        self.slip_model = slip_model
        self.target_to_state_generator = target_to_state_generator
        self.traj_weights = traj_weights
        self.degree = poly_degree
        self.debug = debug
        self.qp_grid_points = qp_grid_points
        self.qp = None
        self.qp_params = {}
        self.qp_vars = {}
        self._last_TO_time = 0.0
        self.max_flight_theta_dot = max_flight_theta_dot
        # This limitation on GRF comes from biomechanical studies of animals and their correspondent natural GRF.
        # Refer to: Blickhan & Full (1993)-Similarity in multilegged locomotion: bouncing like a monopode-fig4-A
        self._max_z_acc = 3.5 * g
        self._max_y_acc = 3.5 * g
        print('Constructing QP (DCP)')
        self.construct_QP_problem()

    def generate_slip_trajectory_tree(self, initial_state, desired_gait_cycles=np.inf, desired_duration=np.inf,
                                      look_ahead_cycles=0, max_samples_per_cycle=20, angle_epsilon=np.deg2rad(1),
                                      verbose=True) -> TrajectoryTreeNode:
        assert xor(np.isinf(desired_gait_cycles), np.isinf(
            desired_duration)), 'Provide ONE stopping criteria: `desired_gait_cycles` or `desired_duration`'
        opt_start = time.time()
        self.present_cycle, self.present_time = (1, 0.0)
        self.max_limits_required = desired_gait_cycles
        cycles_limited = not np.isinf(desired_gait_cycles)
        counter = self.present_cycle if cycles_limited else self.present_time
        limit = desired_gait_cycles if cycles_limited else desired_duration
        root_node = TrajectoryTreeNode(gait_cycle=None, parent_node=None, child_nodes={}, is_root=True)
        root_node._init_to_state = initial_state
        present_node = root_node
        pbar = tqdm(total=(float(limit)), desc='Traj Optimization', dynamic_ncols=True, disable=(not verbose),
                    position=0,
                    leave=True,
                    file=(sys.stdout))
        while counter <= limit:
            self.grow_trajectory_tree(present_node, look_ahead_cycles=look_ahead_cycles,
                                      min_angle_delta=angle_epsilon,
                                      max_samples=max_samples_per_cycle)
            present_node.freeze()
            present_node = present_node.optimal_child_node
            self.present_time = float(present_node.gait_cycle.end_time)
            self.present_cycle += 1
            if cycles_limited:
                counter += 1
                pbar.update()
            else:
                pbar.update(self.present_time - counter)
                counter = self.present_time

        pbar.close()
        return root_node

    def sample_new_angles(self, parent_node: TrajectoryTreeNode, min_angle_delta):
        planned_trajectories = parent_node.child_nodes
        new_angles = []
        if len(planned_trajectories) == 0:
            x_to, xdot_to, _, z_to, zdot_to, _ = parent_node.final_to_state
            z_apex = z_to + zdot_to ** 2 / (2 * g)
            # Sample TD angles with natural TD velocity vector angle with the horizontal (Beta)
            # Refer to: Blickhan & Full (1993)-Similarity in multilegged locomotion: bouncing like a monopode-fig4-A
            beta = np.linspace(np.deg2rad(0), np.deg2rad(35), 20)
            t_fall = np.tan(beta) * xdot_to / g
            z_td = z_apex - 0.5 * g * t_fall ** 2
            # Filter out unfeasible angles
            z_td = z_td[(z_td < self.slip_model.r0)]

            new_angles = np.pi - np.arcsin(z_td / self.slip_model.r0)
            new_angles = new_angles[(MIN_TD_ANGLE <= new_angles)]
            new_angles = new_angles[(new_angles <= MAX_TD_ANGLE)]
            if len(new_angles) == 0:
                new_angles = np.linspace(MIN_TD_ANGLE, MAX_TD_ANGLE, 15)
        else:
            angles = np.array(list(planned_trajectories.keys()))
            costs = [child_node.compute_opt_cost() for child_node in planned_trajectories.values()]
            if np.all(np.isnan(costs)):
                return []
            min_cost = np.nanmin(costs)
            if np.isinf(min_cost):
                min_cost_idx = [i for i in range(len(angles)) if i % 2 != 0]
            else:
                min_cost_idx = [idx for idx, cost in enumerate(costs) if not np.isinf(cost) if
                                abs(cost - min_cost) / cost < 0.1]
            angle_candidates = angles[min_cost_idx]
            sorted_angles = sorted(angles)
            new_angles = []
            for candiate in angle_candidates:
                candidate_idx = sorted_angles.index(candiate)
                if candidate_idx != 0 and candidate_idx != len(sorted_angles) - 1:
                    # Spawn left
                    step = (sorted_angles[candidate_idx] - sorted_angles[(candidate_idx - 1)]) / 2
                    new_left_candidate = sorted_angles[candidate_idx] - step
                    if step > min_angle_delta:
                        new_angles.append(float(new_left_candidate))
                    # Spawn right
                    step = (sorted_angles[(candidate_idx + 1)] - sorted_angles[candidate_idx]) / 2
                    new_right_candidate = sorted_angles[candidate_idx] + step
                    if step > min_angle_delta:
                        new_angles.append(float(new_right_candidate))
                else:
                    if candidate_idx == 0:
                        if sorted_angles[candidate_idx] != MIN_TD_ANGLE:
                            new_angles.append(max([sorted_angles[0] - np.deg2rad(5), MIN_TD_ANGLE]))
                if candidate_idx != len(sorted_angles) - 1:
                    if sorted_angles[(-1)] != MAX_TD_ANGLE:
                        new_angles.append(min([sorted_angles[(-1)] + np.deg2rad(5), MAX_TD_ANGLE]))
                sorted_angles = sorted(sorted_angles + new_angles)

        if np.any(new_angles > MAX_TD_ANGLE) or np.any(new_angles < MIN_TD_ANGLE):
            raise Exception('Sampled touch-down angles out of bounds (min:%.2f, max:%.2f): %s : ' % (
                np.rad2deg(MIN_TD_ANGLE), np.rad2deg(MAX_TD_ANGLE), np.rad2deg(new_angles)))
        return new_angles

    def grow_trajectory_tree(self, parent_opt_node: TrajectoryTreeNode, look_ahead_cycles=0,
                             gait_aux_variables: Optional[Dict] = None, min_angle_delta=np.deg2rad(1),
                             max_samples=15) -> None:
        if gait_aux_variables is None:
            gait_aux_variables = {}

        t_cycle_start = parent_opt_node.gait_cycle.end_time if not parent_opt_node.is_root else 0.0
        expected_to_time = 1 / self.slip_model.spring_natural_freq + t_cycle_start
        cycle = parent_opt_node.cycle_number + 1
        # Update target state generator auxiliary variables.
        gait_aux_variables.update({'cycle': cycle, 't_to': expected_to_time})

        # Limit future planning when the requested limit is reached.
        look_ahead_cycles = min(look_ahead_cycles, self.max_limits_required - cycle)
        n_samples = len(parent_opt_node.child_nodes)

        # If some angles were already planned, and future planning is desired, extend these nodes.
        for theta_td, existing_node in parent_opt_node.child_nodes.items():
            if not np.isinf(existing_node.gait_cycle_optimization_cost) and look_ahead_cycles > 0:
                self.grow_trajectory_tree(parent_opt_node=existing_node, gait_aux_variables={'cycle_in_future': True},
                                          look_ahead_cycles=(look_ahead_cycles - 1),
                                          min_angle_delta=min_angle_delta)

        # Generate new samples if needed.
        while n_samples < max_samples:
            # Spawn new touch down angle samples
            new_td_angles = self.sample_new_angles(parent_opt_node, min_angle_delta)
            if len(new_td_angles) == 0:
                break
            # Compute trajectories for each td angle and generate new tree nodes for each of them.
            for theta_td in new_td_angles:
                x_coeff, z_coeff = self.get_warm_start_coefficients(parent_opt_node, float(theta_td))
                # Create a new branch of the trajectory tree, considering a new td angle for the current gait_cycle.
                gait_aux_variables['cycle_in_future'] = False
                gait_cycle = self.gait_cycle_optimization(to_init_state=parent_opt_node.final_to_state,
                                                          theta_td=np.array(theta_td),
                                                          prev_to_angle=parent_opt_node.final_to_angle,
                                                          gait_aux_variables=gait_aux_variables,
                                                          prev_x_coeff=x_coeff,
                                                          prev_z_coeff=z_coeff)
                gait_cycle.offset_initial_time(t_cycle_start)
                # Create tree node with the resultant gait cycle
                child_node = TrajectoryTreeNode(gait_cycle=gait_cycle, child_nodes={}, parent_node=parent_opt_node)
                # If future planning is required grow tree from node in the future.
                if not np.isinf(gait_cycle.optimization_cost):
                    if look_ahead_cycles > 0:
                        self.grow_trajectory_tree(parent_opt_node=child_node,
                                                  gait_aux_variables={'cycle_in_future': True},
                                                  look_ahead_cycles=(look_ahead_cycles - 1),
                                                  min_angle_delta=min_angle_delta)
                # Introduce node into the parent node children
                parent_opt_node.child_nodes[float(theta_td)] = child_node
                n_samples += 1

    def gait_cycle_optimization(self, to_init_state, theta_td, prev_to_angle, gait_aux_variables: Dict,
                                prev_x_coeff=None, prev_z_coeff=None) -> Union[(SlipGaitCycle, SlipGaitCycleCtrl)]:
        assert to_init_state.shape[0] == 6, 'Invalid Initial TO cartesian State Shape [%s] ' % str(to_init_state.shape)
        assert MAX_TD_ANGLE >= theta_td >= MIN_TD_ANGLE, 'Invalid Touch Down (TD) angle %.3f[Deg]' % np.rad2deg(theta_td)
        td_state_cartesian, tof, x_foot_TD_pos = self.slip_model.predict_td_state(to_init_state, theta_td)
        TD_state_polar, u_TD = self.slip_model.cartesian_to_polar(td_state_cartesian, foot_contact_pos=x_foot_TD_pos)
        # Get flight trajectory
        t_flight = np.linspace(start=0.0, stop=tof, num=10)
        flight_cart_traj = self.slip_model.get_flight_trajectory(t=t_flight, take_off_state=to_init_state)

        flight_theta_dot = np.abs(theta_td - prev_to_angle) / tof if tof != 0.0 else np.inf
        if flight_theta_dot > self.max_flight_theta_dot or TD_state_polar[R] < self.slip_model.r0 * 0.98:
            return SlipGaitCycle(slip_model=self.slip_model, t_flight=t_flight,
                                 flight_cartesian_traj=flight_cart_traj, t_stance=(np.array([tof])),
                                 stance_polar_traj=(np.expand_dims(TD_state_polar, -1)),
                                 optimization_cost=(np.Inf))

        # Simulate the Passive Stance dynamics after Touch Down
        expected_stance_duration = 1 / self.slip_model.spring_natural_freq / 2.0
        t_stance, stance_passive_polar_traj = self.slip_model.get_stance_trajectory(TD_state_polar)
        stance_time = t_stance[(-1)]
        t_stance = np.array(t_stance) + tof
        u_passive = np.zeros((2, t_stance.shape[0]))
        # Get the cartesian stance trajectory used by the QP
        stance_passive_cart_traj = self.slip_model.polar_to_cartesian(trajectory=stance_passive_polar_traj,
                                                                      control_signal=u_passive,
                                                                      foot_contact_pos=0.0)

        if stance_passive_cart_traj[(Z_DOT, -1)] < 0.0 or stance_passive_cart_traj[(X_DOT, -1)] < 0.0 or \
                stance_time < expected_stance_duration * 0.5:
            invalid_cycle = SlipGaitCycle(slip_model=self.slip_model, t_flight=t_flight,
                                          flight_cartesian_traj=flight_cart_traj, t_stance=t_stance,
                                          stance_polar_traj=stance_passive_polar_traj)
            return invalid_cycle
        # Generate target take-off state
        to_des_state = self.target_to_state_generator.get_target_cartesian_state(td_state=td_state_cartesian,
                                                                                 prev_to_state=to_init_state,
                                                                                 stance_time=stance_time,
                                                                                 **gait_aux_variables)
        # Generate target take-off state error weights
        to_des_state_weights = self.target_to_state_generator.get_target_state_weights(**gait_aux_variables)

        # Find the evenly spaced qp_grid_points on the stance dynamics trajectory to function as control points
        # in the quadratic program optimization
        coarse_stance_passive_cart_traj = stance_passive_cart_traj
        coarse_t_stance = t_stance
        if self.qp_grid_points < stance_passive_cart_traj.shape[1]:
            coarse_idx = np.round(np.linspace(0, stance_passive_cart_traj.shape[1] - 1,
                                              self.qp_grid_points)).astype(int)
            coarse_stance_passive_cart_traj = stance_passive_cart_traj[:, coarse_idx]
            coarse_t_stance = t_stance[coarse_idx]

        # Do Quadratic Program Optimization ___________________________________________________________________________
        qp_start = time.time()
        td_init_state = td_state_cartesian
        td_init_state[X] -= x_foot_TD_pos
        x_coeff, z_coeff, opt_cost = self.solve_QP(to_des_state=to_des_state, to_des_state_weights=to_des_state_weights,
                                                   td_init_state=td_init_state,
                                                   passive_trajectory=coarse_stance_passive_cart_traj,
                                                   traj_times=(coarse_t_stance - t_stance[0]),
                                                   prev_x_coeff=prev_x_coeff,
                                                   prev_z_coeff=prev_z_coeff)
        qp_end = time.time()
        if np.isinf(opt_cost):
            invalid_cycle = SlipGaitCycle(slip_model=(self.slip_model), t_flight=t_flight,
                                          flight_cartesian_traj=flight_cart_traj,
                                          t_stance=t_stance,
                                          stance_polar_traj=stance_passive_polar_traj)
            return invalid_cycle

        stance_controlled_cart_traj = self.get_optimal_trajectory(x_coeff, z_coeff, times=(t_stance - t_stance[0]))
        stance_controlled_cart_traj = np.array(stance_controlled_cart_traj)
        stance_controlled_polar_traj, u_optimal = self.slip_model.cartesian_to_polar(stance_controlled_cart_traj,
                                                                                     foot_contact_pos=0.0)
        if self.debug:
            print('\n** \t Stance Differentially Flat Trajectory Optimization \t **')
            print('Touch Down (TD) Angle: %.3f[Deg]' % (theta_td * 180 / np.pi))
            print('QP overall new_start_time: %.4f[s]' % (qp_end - qp_start))
            print('QP reported new_start_time: %.4f[s]' % self.qp.solver_stats.solve_time)
            print('QP optimal cost: %.4f[s]' % opt_cost)
            print('Estimate flight new_start_time %.2f [s]' % tof)
            print('Flight_Trajectory (cartesian): ', flight_cart_traj.shape)
            print('Stance Passive Trajectory (cartesian) Shape:', stance_passive_cart_traj.shape)
            print('Stance Controlled Trajectory (cartesian) Shape:', stance_controlled_cart_traj.shape)
            print('Stance Coarse Controlled Trajectory (cartesian) Shape:', coarse_stance_passive_cart_traj.shape)

        return SlipGaitCycleCtrl(slip_model=(self.slip_model), optimization_cost=opt_cost, t_flight=t_flight,
                                       flight_cartesian_traj=flight_cart_traj,
                                       t_stance=t_stance,
                                       stance_ctrl_polar_traj=stance_controlled_polar_traj,
                                       control_signal=u_optimal,
                                       stance_passive_polar_traj=stance_passive_polar_traj,
                                       target_to_state=to_des_state,
                                       ctrl_kwargs={'x_coeff': x_coeff,
                                                    'z_coeff': z_coeff})

    def solve_QP(self, to_des_state, to_des_state_weights, td_init_state, passive_trajectory, traj_times,
                 prev_x_coeff=None, prev_z_coeff=None):
        assert passive_trajectory.shape[0] == to_des_state.shape[0] == 6, "[x, x', x'', z, z', z'']"
        num_ref_traj_points = traj_times.shape[0]
        assert self.qp_grid_points >= num_ref_traj_points > 0, 'Invalid passive trajectory length'

        # Retrieve names of QP parameters and values
        x_coeff_var, z_coeff_var = self.qp_vars['x_coeff'], self.qp_vars['z_coeff']
        phi_diag_params = self.qp_params['phi_diag']
        Q_max_sqrt_params = self.qp_params['Q_max_sqrt']
        B_params = self.qp_params['B']
        C_params = self.qp_params['C']
        x_ref_init_param = self.qp_params['x_ref_init']

        # Set up grid_point invariant values
        Q = np.diag(self.traj_weights.astype(np.float64))
        Q_final = np.diag(to_des_state_weights.astype(np.float64))
        x_ref_init_param.value = np.expand_dims(td_init_state, 1)  # TD state constraint

        # Fixed first 3 coefficients of both polynomials to the TD event values at stance time 0.
        x_coeff_init = np.zeros((self.degree + 1, 1))
        x_coeff_init[:3, 0] = [td_init_state[X], td_init_state[X_DOT], td_init_state[X_DDOT] / 2]
        z_coeff_init = np.zeros((self.degree + 1, 1))
        z_coeff_init[:3, 0] = [td_init_state[Z], td_init_state[Z_DOT], td_init_state[Z_DDOT] / 2]
        if prev_x_coeff is not None:
            if prev_z_coeff is not None:
                x_coeff_init[3:, 0] = prev_x_coeff[3:, 0]
                z_coeff_init[3:, 0] = prev_z_coeff[3:, 0]
        x_coeff_var.value = x_coeff_init
        z_coeff_var.value = z_coeff_init

        # Compute empirical delta times
        delta_times = traj_times[1:] - traj_times[:-1]
        for ref_grid_point in range(num_ref_traj_points):
            t = traj_times[ref_grid_point]
            # Calculate Phi Matrix [Phi_1, Phi_2, Phi_3]
            phi = SlipDiffFlatController.get_polynomial_basis(degree=(self.degree), t=t)
            phi_diag = block_diag(phi, phi)
            phi_diag_params[ref_grid_point].value = phi_diag

            if ref_grid_point == num_ref_traj_points - 1:  # Take off
                x_ref = np.expand_dims(to_des_state, 1)
                QQ = Q_final
            else:
                x_ref = passive_trajectory[:, [ref_grid_point]]
                dt = delta_times[ref_grid_point]
                QQ = Q * dt

            # Update Quadratic Symmetric Positive Semi-definite matrix.
            Q_max = phi_diag.T @ QQ @ phi_diag
            # Compute Square Root Of Q_max
            eigen_values, left_eigen_vectors = scipy.linalg.eigh(Q_max)
            eigen_values[eigen_values < 0] = 0
            # Numerically stable way of obtaining sqrt(Q_max)
            Q_max_sqrt = left_eigen_vectors * np.sqrt(eigen_values) @ left_eigen_vectors.T
            # Quadratic Cost: x_ref.T @ phi_diag.T @ Q @ phi_diag @ y_ref --> lambd.T @ Q_max @ lambd
            Q_max_sqrt_params[ref_grid_point].value = Q_max_sqrt
            # Linear Cost: -2 * y_ref.T @ Q @ phi_diag @ lambd
            B_params[ref_grid_point].value = -2 * x_ref.T @ QQ @ phi_diag
            # Constant Term: c = y_ref.T @ QQ @ y_ref --> b    # Can be ignored.
            C_params[ref_grid_point].value = x_ref.T @ QQ @ x_ref

        if num_ref_traj_points < self.qp_grid_points:
            # Reference trajectory was shorter than the number of QP grid points.
            # Make the solver ignore all missing grid points.
            for missing_grid_point in range(num_ref_traj_points, self.qp_grid_points - 1):
                phi_diag_params[missing_grid_point].value = np.zeros(phi_diag_params[missing_grid_point].shape)
                B_params[missing_grid_point].value = np.zeros(B_params[missing_grid_point].shape)
                C_params[missing_grid_point].value = np.zeros(C_params[missing_grid_point].shape)
                Q_max_sqrt_params[missing_grid_point].value = np.zeros(Q_max_sqrt_params[missing_grid_point].shape)

        self.qp.solve(solver='OSQP', verbose=(self.debug), warm_start=True, polish=False, time_limit=0.05)
        return x_coeff_var.value, z_coeff_var.value, self.qp.value

    def construct_QP_problem(self):
        """
        Constructs the CVXPX `Problem` instance, complying with the Disciplined Parameterized Programming DPP, the
        problem is built using cvxpy `Parameter` class to generate a template of the overall optimization process, i.e.
        building a symbolic computational graph, whose parameters will be appropiatedly updated during the recurrent
        optimization process gaining almost an order of magnitude reduce in solving slapsed new_start_time.
        The QP cost function had to be parameterized to comply with DPP, which might complicate the understanding of the
        problem setup, however have a paralel look at the function `solve_QP` to understand the process.
        Helpfull Links: https://www.cvxpy.org/tutorial/advanced/index.html#disciplined-parametrized-programming
        See for details: Learning to run naturally: guiding policies with the Spring-Loaded Inverted Pendulum
        """
        x_coeff_var = cp.Variable(shape=(self.degree + 1, 1), name='x_coeff')
        z_coeff_var = cp.Variable(shape=(self.degree + 1, 1), name='z_coeff')

        lambd = cp.vstack((x_coeff_var, z_coeff_var))  # Variable of polynomial coefficients of x and z dims
        # max_r = cp.Constant(self.slip_model.r0)
        gravity = cp.Constant(g)
        TO_z_dot_min = cp.Constant(0.01)   # Take off vertical vel
        x_ref_init_param = cp.Parameter(shape=(6, 1), name='x_ref_init')  # td state
        Q_max_sqrt_params = []
        phi_diag_params = []
        B_params = []
        C_params = []

        total_cost = 0.0
        constraints = []
        # Impose quadratic error cost on each of the node_points
        # Add costs at specific points on the desired trajectory
        for grid_point in range(self.qp_grid_points):
            # Parameter hoolding the Q_max matrix for the specific grid point
            Q_max_sqrt = cp.Parameter(shape=(2 * (self.degree + 1), 2 * (self.degree + 1)), PSD=True,
                                      name=('Q_max_sqrt_%d' % grid_point))
            Q_max_sqrt.value = np.zeros(Q_max_sqrt.shape)
            # Parameter holding the Phi variable for the specific grid point new_start_time (`t`)
            phi_diag_param = cp.Parameter(shape=(6, 2 * (self.degree + 1)), name=('phi_diag_%d' % grid_point))
            phi_diag_param.value = np.zeros(phi_diag_param.shape)
            # Allocate resultant quadratic `b` matrix value
            # b = y_ref.T @ Q @ phi_diag
            b = cp.Parameter(shape=(1, 2 * (self.degree + 1)), name=('x_ref.Q.phi_%d' % grid_point))
            b.value = np.zeros(b.shape)
            # Constant value in the quadratic error cost is ignored as it does not depend on the polynomial parameters
            # and its computation is ommited for efficiency.
            # c = y_ref_param.T @ QQ @ y_ref_param
            c = cp.Parameter(shape=(1, 1), name=('C_%d' % grid_point))
            c.value = np.zeros(c.shape)
            # Total Cost Equation
            # Quadratic Cost: y_ref.T @ phi_diag.T @ Q @ phi_diag @ y_ref --> lambd.T @ Q_max @ lambd
            # Linear Cost: -2 * y_ref.T @ Q @ phi_diag @ lambd            --> b @ lambd
            total_cost += cp.sum_squares(Q_max_sqrt @ lambd) + b @ lambd + c

            if grid_point == 0:
                # Add linear constraint fixing the starting point of the optimal slip_traj
                # Constraint Formula: A*ref_state = b   --> vars = ref_state
                constraints.append(phi_diag_param @ lambd == x_ref_init_param)
            else:
                if grid_point == self.qp_grid_points - 1:
                    phi_1 = phi_diag_param[[1], :self.degree + 1]
                    phi_2 = phi_diag_param[[2], :self.degree + 1]
                    constraints.append(phi_1 @ z_coeff_var >= TO_z_dot_min)
                    constraints.append(phi_2 @ z_coeff_var == -gravity)
                    constraints.append(phi_2 @ x_coeff_var == 0.0)
                else:
                    phi_2 = phi_diag_param[[2], :self.degree + 1]
                    constraints.append(phi_2 @ x_coeff_var <= self._max_y_acc)
                    constraints.append(phi_2 @ z_coeff_var <= self._max_z_acc)
            phi_diag_params.append(phi_diag_param)
            Q_max_sqrt_params.append(Q_max_sqrt)
            B_params.append(b)
            C_params.append(c)

        prob = cp.Problem(objective=cp.Minimize(total_cost), constraints=constraints)
        assert prob.is_dcp(dpp=True), "Problem is not DPP, recursive solving will not be efficient"
        assert prob.is_dcp(dpp=False), "Problem is not DCP, it will either be infeasible or solved with a wrong solver"

        self.qp = prob
        self.qp_vars = {'x_coeff': x_coeff_var, 'z_coeff': z_coeff_var}
        self.qp_params = {'phi_diag': phi_diag_params, 'Q_max_sqrt': Q_max_sqrt_params,
                          'x_ref_init': x_ref_init_param,
                          'B': B_params,
                          'C': C_params}

    def get_optimal_trajectory(self, x_coeff, z_coeff, times):
        lambd = np.vstack((x_coeff, z_coeff))
        degree = x_coeff.shape[0] - 1
        pred_traj = np.zeros((6, times.shape[0]))
        for i, t in enumerate(times):
            phi = SlipDiffFlatController.get_polynomial_basis(degree=degree, t=t)
            phi_diag = block_diag(phi, phi)
            pred_traj[:, i] = np.squeeze(phi_diag.dot(lambd))

        return pred_traj

    @staticmethod
    def get_warm_start_coefficients(parent_trajectory_node, theta_td) -> [np.ndarray, np.ndarray]:
        x_coeff, z_coeff = (None, None)
        planned_angles = np.array([k for k, node in parent_trajectory_node.child_nodes.items() if
                                   not np.isinf(node.gait_cycle_optimization_cost)])
        if len(planned_angles) != 0:
            closest_angle_idx = np.argmin(np.abs(planned_angles - theta_td))
            gait_cycle = parent_trajectory_node.child_nodes[planned_angles[closest_angle_idx]].gait_cycle
        else:
            gait_cycle = parent_trajectory_node.gait_cycle
        if isinstance(gait_cycle, SlipGaitCycleCtrl):
            if 'x_coeff' in gait_cycle.ctrl_kwargs:
                x_coeff, z_coeff = gait_cycle.ctrl_kwargs['x_coeff'], gait_cycle.ctrl_kwargs['z_coeff']
        return (
            x_coeff, z_coeff)

    @staticmethod
    def get_polynomial_basis(degree, t):
        """
        This method returns the polynomial basis of n degree of the cero, first and second derivatives
    
        phi_0 = [1, t, t**2, t**3, ..., t**n]
        phi_1 = [0, 1, 2t, 3*t**2, ..., n*t**(n-1)]
        phi_2 = [0, 0, 2, 2*3**t, ..., n*(n-1)*t**(n-2)]
    
        """
        phi_0 = np.zeros((degree + 1), dtype=(np.float64))
        phi_1 = np.zeros((degree + 1), dtype=(np.float64))
        phi_2 = np.zeros((degree + 1), dtype=(np.float64))
        for i in range(0, degree + 1):
            phi_0[i] = t ** i

        coeff = np.arange(1, degree + 1)
        phi_1[1:] = phi_0[:-1] * coeff
        phi_2[2:] = phi_0[:-2] * coeff[1:]
        phi_2[3:] *= coeff[1:degree - 1]
        return np.vstack((phi_0, phi_1, phi_2))
