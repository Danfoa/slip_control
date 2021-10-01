# -*- coding: utf-8 -*-
# @Author  : Daniel Ordonez
# @email   : daniels.ordonez@gmail.com

from typing import Optional

import numpy as np
from numpy import ndarray

from .slip_model import SlipModel, THETA, X


class SlipGaitCycle(object):

    def __init__(self, slip_model: SlipModel, t_flight: ndarray, flight_cartesian_traj: ndarray, t_stance: ndarray,
                 stance_polar_traj: ndarray, target_to_state: Optional[ndarray] = None,
                 optimization_cost: Optional[float] = np.NaN):
        """
        Class representing a Slip gait cycle composed of:
         - a flight phase: starting with the take-off (TO) event of a previous gait cycle and ending at the present cycle
          touch-down (TD) event. And,
         - a stance phase: starting at the present cycle TD event and ending with the cycle's TO event.
        It is assumed the gait cycle starts with a flight phase and ends with the stance phase.
        :param slip_model: SLIP model providing the `m`, `k` and `r0` parameters.
        :param t_flight: (F,) Discrete time array during flight phase
        :param flight_cartesian_traj: (6, F) Cartesian state [x, x', x'', z, z', z''] of the SLIP CoM at each time
        during the flight phase
        :param t_stance: (S,) Discrete time array during flight phase
        :param stance_polar_traj: (4, S) Polar state [theta,theta',r,r'] of the SLIP CoM at each time during
        the stance phase, in a reference frame centered at the foot contact point with the ground.
        :param target_to_state: (6,) Optional cartesian target take-off state, used for control. Use np.NaN in the
        dimensions of the cartesian state that are irrelevant for control (e.g. z''=np.NaN)
        :param optimization_cost: Optional scalar indicating the optimization cost of the gait cycle. By default is set 
        to be the euclidean norm of the error between the real TO state and the target TO state (ignoring np.NaNs dims) 
        """
        self.slip_model = slip_model
        if not flight_cartesian_traj.shape[0] == 6:
            raise AssertionError("Expected cartesian trajectory [x, x', x'', z, z', z'']")
        else:
            if not len(t_flight) == flight_cartesian_traj.shape[1]:
                raise AssertionError('Invalid flight trajectory')
            else:
                self.t_flight = t_flight
                self.flight_cartesian_traj = flight_cartesian_traj
                assert stance_polar_traj.shape[0] == 4, "Expected stance polar trajectory [theta, theta', r, r']"
                assert len(t_stance) == stance_polar_traj.shape[1], 'Invalid flight trajectory'
            self.t_stance = t_stance
            self.stance_polar_traj = stance_polar_traj
            self._stance_cartesian_traj = None
            self.touch_down_angle = self.stance_polar_traj[(THETA, 0)]
            self.take_off_angle = self.stance_polar_traj[(THETA, -1)]
            self.foot_contact_pos = self.flight_cartesian_traj[(X, -1)] - self.slip_model.r0 * np.cos(
                self.touch_down_angle)
            assert self.t_flight[(-1)] == self.t_stance[
                0], 'Touch down state should be in flight and stance trajectories'
        self.target_to_state = np.array(target_to_state) if target_to_state is not None else np.ones((6,)) * np.NaN
        self.optimization_cost = optimization_cost
        if not np.all(np.isnan(self.target_to_state)):
            self.target_to_state = np.ma.array((self.target_to_state), mask=(np.isnan(self.target_to_state)))
            if np.isnan(optimization_cost):
                self.optimization_cost = np.linalg.norm(self.take_off_state - target_to_state)

    @property
    def stance_cartesian_traj(self) -> ndarray:
        if self._stance_cartesian_traj is None:
            self._stance_cartesian_traj = self.slip_model.polar_to_cartesian(trajectory=(self.stance_polar_traj),
                                                                             foot_contact_pos=(self.foot_contact_pos))
        return self._stance_cartesian_traj

    @property
    def take_off_state(self) -> ndarray:
        if self._stance_cartesian_traj is None:
            return self.slip_model.polar_to_cartesian((self.stance_polar_traj[:, -1]),
                                                      foot_contact_pos=(self.foot_contact_pos))
        else:
            return np.array(self.stance_cartesian_traj[:, -1])

    @property
    def take_off_state_polar(self):
        return np.array(self.stance_polar_traj[:, -1])

    @property
    def touch_down_state(self):
        if self._stance_cartesian_traj is None:
            return self.slip_model.polar_to_cartesian((self.stance_polar_traj[:, 0]),
                                                      foot_contact_pos=(self.foot_contact_pos))
        else:
            return np.array(self.stance_cartesian_traj[:, 0])

    @property
    def touch_down_state_polar(self):
        return np.array(self.stance_polar_traj[:, 0])

    @property
    def prev_take_off_state(self):
        return np.array(self.flight_cartesian_traj[:, 0])

    @property
    def start_time(self):
        return float(self.t_flight[0])

    @property
    def end_time(self):
        return float(self.t_stance[(-1)])

    def offset_initial_time(self, time_offset):
        self.t_flight += time_offset
        self.t_stance += time_offset

    def __str__(self):
        return 'Cost:%.2f TD:%.1f[deg] TO:%.1f[deg] time:[%.2f, %.2f]' % (
            self.optimization_cost, np.rad2deg(self.touch_down_angle), np.rad2deg(self.take_off_angle),
            self.start_time, self.end_time)

    def __repr__(self):
        return str(self)


class SlipGaitCycleCtrl(SlipGaitCycle):

    def __init__(self, slip_model, t_flight, flight_cartesian_traj, t_stance, stance_passive_polar_traj=None,
                 stance_ctrl_polar_traj=None, control_signal=None, target_to_state=None, optimization_cost=np.NaN,
                 ctrl_kwargs=None):
        """
        Class representing a Controlled Slip gait Cycle. This assumes the SLIP model is an actuated extended version
        (see "Learning to run naturally: Guiding policies with the Spring-Loaded Inverted Pendulum" Chap 4.1) where the
        control inputs are a resting leg length displacement `r_delta` (axial force control) and a hip torque `tau_hip`.
        The main difference with the parent class it that `SlipGaitCycleCtrl` stores also a `control_signal` and if
        provided an additional `SlipGaitCycle` instance representing the passive dynamical response of SLIP (useful for
        plotting animation and intuition).
        :param slip_model: SLIP model providing the `m`, `k` and `r0` parameters.
        :param t_flight: (F,) Discrete time array during flight phase
        :param flight_cartesian_traj: (6, F) Cartesian state [x,x',x'',z,z',z''] of the SLIP CoM at each time during
        the flight phase
        :param t_stance: (S,) Discrete time array during flight phase
        :param stance_passive_polar_traj: (4, S) Passive Polar state [theta,theta',r,r'] of the SLIP CoM at each time 
        during the stance phase, in a reference frame centered at the foot contact point with the ground.
        :param stance_ctrl_polar_traj: (4, S) Controlled Polar state [theta,theta',r,r'] of the SLIP CoM at each time 
        during the stance phase, in a reference frame centered at the foot contact point with the ground.
        :param control_signal: (2, S) Control input signal assumed to hold ()
        :param target_to_state: (6,) Optional cartesian target take-off state, used for control. Use np.NaN in the
        dimensions of the cartesian state that are irrelevant for control (e.g. z''=np.NaN)
        :param optimization_cost: Optional scalar indicating the optimization cost of the gait cycle. By default is set
        to be the euclidean norm of the error between the real TO state and the target TO state (ignoring np.NaNs dims)
        :param ctrl_kwargs: Dictionary holding controller-related keyword arguments.
        """
        self.control_signal = control_signal
        self.optimization_cost = optimization_cost
        self.control_signal = control_signal
        self.ctrl_kwargs = ctrl_kwargs
        super(SlipGaitCycleCtrl, self).__init__(slip_model, t_flight, flight_cartesian_traj, t_stance,
                                                stance_polar_traj=stance_ctrl_polar_traj,
                                                target_to_state=target_to_state)
        if stance_passive_polar_traj is not None:
            self.passive_gait_cycle = SlipGaitCycle(slip_model, t_flight, flight_cartesian_traj, t_stance,
                                                    stance_polar_traj=stance_passive_polar_traj)
        else:
            self.passive_gait_cycle = None

    @property
    def stance_cartesian_traj(self) -> ndarray:
        if self._stance_cartesian_traj is None:
            self._stance_cartesian_traj = self.slip_model.polar_to_cartesian(trajectory=(self.stance_polar_traj),
                                                                             control_signal=(self.control_signal),
                                                                             foot_contact_pos=(self.foot_contact_pos))
        return self._stance_cartesian_traj

    @property
    def take_off_state(self) -> ndarray:
        if self._stance_cartesian_traj is None:
            return self.slip_model.polar_to_cartesian((self.stance_polar_traj[:, -1]),
                                                      control_signal=(self.control_signal[:, -1]),
                                                      foot_contact_pos=(self.foot_contact_pos))
        else:
            return np.array(self.stance_cartesian_traj[:, -1])

    @property
    def touch_down_state(self):
        if self._stance_cartesian_traj is None:
            return self.slip_model.polar_to_cartesian((self.stance_polar_traj[:, 0]),
                                                      control_signal=(self.control_signal[:, 0]),
                                                      foot_contact_pos=(self.foot_contact_pos))
        else:
            return np.array(self.stance_cartesian_traj[:, 0])
