import copy
import pathlib
from typing import Union, List

import numpy as np
import pickle5 as pickle
from numpy import ndarray
from scipy.interpolate import interp1d

from .slip_gait_cycle import SlipGaitCycle
from .slip_model import SlipModel, THETA, X, THETA_DOT, X_DOT


class SlipTrajectory:
    FILE_EXTENSION = '.slip_traj'

    def __init__(self, slip_model: SlipModel, slip_gait_cycles: Union[(List[SlipGaitCycle], SlipGaitCycle)]):
        self.slip_model = slip_model
        self.gait_cycles = slip_gait_cycles if isinstance(slip_gait_cycles, List) else [slip_gait_cycles]
        self._continuous_gait_phase, self._continuous_target_forward_vel = (None, None)
        self._continuous_polar_traj, self._continuous_cart_traj = (None, None)

    def append(self, other: Union[(SlipGaitCycle, 'SlipTrajectory')]):
        """
        Utility function to append another `SlipTrajectory` or `SlipGaitCycle` at the end of the current trajectory.
        This function asserts time and state coupling.
        :param other: SlipTrajectory or SlipGaitCycle to append to the present trajectory
        """
        other_init_gait_cycle = other if isinstance(other, SlipGaitCycle) else other.gait_cycles[0]
        SlipTrajectory.assert_continuity_between_cycles(self.gait_cycles[(-1)], other_init_gait_cycle)
        if isinstance(other, SlipGaitCycle):
            self.gait_cycles.append(copy.deepcopy(other))
        else:
            self.gait_cycles.extend(copy.deepcopy(other.gait_cycles))

    @staticmethod
    def assert_continuity_between_cycles(gait_cycle_1: SlipGaitCycle, gait_cycle_2: SlipGaitCycle):
        MIN_DT = 0.01
        dt = gait_cycle_2.start_time - gait_cycle_1.end_time
        if not (dt >= 0 and dt <= MIN_DT):
            raise AssertionError('Traj times dont align %.3f, %.3f' % (gait_cycle_1.end_time,
                                                                       gait_cycle_2.start_time))
        else:
            state_diff = gait_cycle_1.take_off_state - gait_cycle_2.prev_take_off_state
            # assert np.allclose(state_diff, 0, rtol=0.1, atol=0.1), state_diff

    @property
    def optimization_cost(self):
        return sum([cycle.optimization_cost for cycle in self.gait_cycles])

    @property
    def start_time(self) -> float:
        if len(self) == 0:
            raise ValueError('Empty trajectory')
        return self.gait_cycles[0].start_time

    @property
    def end_time(self) -> float:
        if len(self) == 0:
            raise ValueError('Empty trajectory')
        return self.gait_cycles[(-1)].end_time

    def set_initial_time(self, new_start_time):
        old_start_time = self.start_time
        for cycle in self.gait_cycles:
            cycle.offset_initial_time(-old_start_time + new_start_time)

    def __len__(self):
        return len(self.gait_cycles)

    def gen_continuous_trajectory(self) -> [
        interp1d, interp1d]:
        """
        Utility function to obtain an interpolator of the entire SLIP trajectory cartesian [x, x', x'', z, z', z'']
        and polar states [theta, theta', r, r'], on its time domain, allowing to get state information at any time
        in [start, end], or a single array containing the trajectory across gait cycles.
        Note: During each flight phase the polar states are filled with polar states mimicking the leg motion from
        previous take-off to the next touch-down states. These states are included for animation purposes.
        :return: Two interpolators one for the cartesian coordinates and the other for polar coordinates
        """
        cart_trajs, polar_trajs, times = [], [], []
        TO_state_polar = np.array([np.pi/2, 0, self.slip_model.r0*0.8, 0])
        TD_state_polar = None
        for cycle in self.gait_cycles:
            cart_trajs.extend([cycle.flight_cartesian_traj, cycle.stance_cartesian_traj])
            times.extend([cycle.t_flight, cycle.t_stance])

            # TO_state_initial = cycle.prev_take_off_state.copy()
            TD_state_polar = cycle.stance_polar_traj[:, 0].copy()
            t_flight_end, t_flight_start = cycle.t_flight[(-1)], cycle.t_flight[0]
            theta_dot_flight = (TO_state_polar[THETA] - TD_state_polar[THETA]) / (t_flight_end - t_flight_start)
            middle_angle = (TD_state_polar[THETA] - TO_state_polar[THETA]) / 2 + TO_state_polar[THETA]

            middle_state = np.array([middle_angle, theta_dot_flight, 0.8 * self.slip_model.r0, 0.0])
            TO_state_polar[THETA_DOT] = theta_dot_flight
            TD_state_polar[THETA_DOT] = theta_dot_flight
            coarse_flight_traj_polar = np.hstack([np.expand_dims(TO_state_polar, axis=1),
                                                 np.expand_dims(middle_state, axis=1),
                                                 np.expand_dims(TD_state_polar, axis=1)])
            flight_duration = t_flight_end - t_flight_start
            flight_polar_traj = interp1d(x=[t_flight_start, t_flight_start + flight_duration / 2, t_flight_end],
                                         y=coarse_flight_traj_polar,
                                         kind='linear',
                                         axis=1,
                                         assume_sorted=True)(cycle.t_flight)

            cycle_polar_trajectories = [flight_polar_traj, cycle.stance_polar_traj]
            polar_trajs.extend(cycle_polar_trajectories)
            # ___
            TO_state_polar = cycle.stance_polar_traj[:, -1]



        final_cart_traj = np.concatenate(cart_trajs, axis=1)
        final_polar_traj = np.concatenate(polar_trajs, axis=1)
        t = np.concatenate(times)
        polar_traj = interp1d(x=t, y=final_polar_traj, axis=1, kind='linear', fill_value='extrapolate',
                              assume_sorted=True)
        cart_traj = interp1d(x=t, y=final_cart_traj, axis=1, kind='linear', fill_value='extrapolate',
                             assume_sorted=True)
        return (cart_traj, polar_traj)

    def gen_continuous_target_forward_velocity(self) -> interp1d:
        target_x_dot, t = [], []
        for cycle in self.gait_cycles:
            target_x_dot.append(cycle.target_to_state[X_DOT])
            t.append(cycle.t_flight[0])

        target_x_dot.append(target_x_dot[(-1)])
        t.append(self.gait_cycles[(-1)].t_stance[(-1)])
        continuous_x_dot_des = interp1d(x=t, y=target_x_dot, kind='linear', fill_value='extrapolate',
                                        assume_sorted=True)
        return continuous_x_dot_des

    def gen_continuous_gait_phase_signal(self) -> interp1d:
        """
        Utility function to obtain a continuous phase signal using linear interpolator of the discrete phase values.
        The phase signal is defined as a linear interpolator in time of `0` to `PI` during a Slip Gait Cycle flight
        phase, and from `PI` to `2PI` during the stance phase.
        :return: (interp1d) One-dim interpolator of the gait phase signal of the `SlipTrajectory`.
        """
        dt = 0.0001
        gait_values, t = [], []
        for cycle in self.gait_cycles:
            touch_down_time = cycle.t_flight[(-1)]
            t.extend([cycle.start_time, touch_down_time, cycle.end_time - dt])

        # t.append(self.end_time)
        gait_cycle_phase = np.linspace(0, 2 * np.pi, 3)
        period_cycle_phase = np.concatenate([gait_cycle_phase] * len(self))
        continuous_gait_phase_signal = interp1d(x=t, y=period_cycle_phase, kind='linear', fill_value='extrapolate',
                                                assume_sorted=True)
        return continuous_gait_phase_signal

    def get_time_signal(self):
        """
        :return: (array) time signal during the entire trajectory, obtained through the unification of the time signals
        of each individual gait cycle
        """
        if len(self) == 0:
            raise ValueError('Empty trajectory')
        t = np.unique(np.concatenate([[cycle.t_flight, cycle.t_stance] for cycle in self.gait_cycles]))
        return t

    def get_state_at_time(self, t: Union[(float, ndarray)]):
        """
        Function to obtain the cartesian [x, x', x'', z, z', z''] and polar states [theta, theta', r, r'] of the CoM
        of the SLIP model at an specific time `t` in the trajectory
        :param t: (ndarray, (float)) Time at which to calculate the gait phase value.
        :return: Cartesian (6, N) and Polar (4, N) states of the SLIP model. N refers to the length of the time vector
        """
        t_gait = t % (self.end_time - self.start_time)
        if self._continuous_cart_traj is None or self._continuous_polar_traj is None:
            cart_traj, polar_traj = self.gen_continuous_trajectory()
            self._continuous_cart_traj = cart_traj
            self._continuous_polar_traj = polar_traj
        return (self._continuous_cart_traj(t_gait), self._continuous_polar_traj(t_gait))

    def get_target_forward_speed(self, t: Union[(float, ndarray)]):
        """
        Evaluate the target forward velocity at an specific time `t`. The target forward velocity is defined to be
        the present gait cycle target take-off state forward velocity. Therefore it changes only at the start of each
        gait cycle.
        :param t: (ndarray, (float)) Time at which to calculate the gait phase value.
        :return: Target forward velocity (1, N). N refers to the length of the time vector
        """
        t_gait = t % (self.end_time - self.start_time)

        if self._continuous_target_forward_vel is None:
            self._continuous_target_forward_vel = self.gen_continuous_target_forward_velocity()
        return self._continuous_target_forward_vel(t_gait)

    def get_gait_phase(self, t: Union[(float, ndarray)]):
        """
        Calculate the SlipTrajectory gait cycle phase value at a given time/s `t`. The phase signal is defined as a
        linear interpolator in time of `0` to `PI` during a Slip Gait Cycle flight phase, and from `PI` to `2PI` during
        the stance phase.
        :param t: (ndarray, (float)) Time at which to calculate the gait phase value.
        :return: Gait phase signal value computed at `t`
        """
        t_gait = t % (self.end_time - self.start_time)

        if self._continuous_gait_phase is None:
            self._continuous_gait_phase = self.gen_continuous_gait_phase_signal()
        phase = self._continuous_gait_phase(t_gait)
        if phase.size == 1:
            phase = float(phase)
        return phase

    @staticmethod
    def save_to_file(traj: 'SlipTrajectory', trajectories_folder, file_name):
        """
        Save a Slip Trajectory to disk using Pickle. The continuous trajectories are not stored in file.
        :param traj: (SlipTrajectory) trajectory to save
        :param trajectories_folder: folder in which trajectories will be stored
        :param file_name:
        :return:
        """
        tmp_traj = copy.deepcopy(traj)
        path = pathlib.Path(trajectories_folder)
        slip_model_folder = 'r0=%.2f_mass=%.2f_k-rel=%.2f' % (tmp_traj.slip_model.r0,
                                                              tmp_traj.slip_model.m,
                                                              tmp_traj.slip_model.k_rel)
        path = path.joinpath(slip_model_folder)
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        file_name = file_name + tmp_traj.FILE_EXTENSION
        path = path.joinpath(file_name)
        tmp_traj._continuous_polar_traj = None
        tmp_traj._continuous_cart_traj = None
        tmp_traj._continuous_target_forward_vel = None
        tmp_traj._continuous_gait_phase = None

        with open(path, 'wb') as (output):
            pickle.dump(tmp_traj, output, pickle.HIGHEST_PROTOCOL)
        print('Trajectory saved to [%s]' % str(path))
        return path

    @staticmethod
    def from_file(file_path) -> 'SlipTrajectory':
        """
        Load a Slip Trajectory saved with pickle in disk
        :param file_path: (str) path to file
        :return: Slip Trajectory stored in file
        """
        path = pathlib.Path(file_path)
        if not path.suffix == SlipTrajectory.FILE_EXTENSION:
            raise AttributeError('File path [%s] must have a %s extension' % (file_path, SlipTrajectory.FILE_EXTENSION))
        with open(path, 'rb') as (input):
            slip_traj = pickle.load(input)
        return slip_traj

    def gen_periodic_traj(self, max_time) -> 'SlipTrajectory':
        """
        Return a slip trajectory composed of periodic repetitions of the original (self) trajectory until the desired
        time duration is reached. Useful for replicating in time a limit-cycle.
        """
        if len(self) == 0:
            raise ValueError('Empty trajectory')
        last_x_pos = float(self.gait_cycles[(-1)].take_off_state[X])
        new_traj = copy.deepcopy(self)
        tmp_traj = copy.deepcopy(self)
        while new_traj.end_time < max_time:
            for cycle in tmp_traj.gait_cycles:
                cycle.flight_cartesian_traj[X, :] += last_x_pos
                cycle.foot_contact_pos += last_x_pos
                cycle._stance_cartesian_traj = None
            tmp_traj.set_initial_time(new_traj.end_time)
            new_traj.append(tmp_traj)

        return new_traj

    def __str__(self):
        return "t[%.2f, %.2f]_cost[%s]" % (self.start_time, self.end_time, self.optimization_cost)