# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.13 |Anaconda, Inc.| (default, Feb 23 2021, 21:15:04) 
# [GCC 7.3.0]
# Embedded file name: /home/dordonez/Projects/slip_control/slip_control/controllers/target_to_states_generator.py
# Compiled at: 2021-09-30 17:51:21
# Size of source mod 2**32: 4853 bytes
import numpy as np
from numpy import ndarray

from ..slip.slip_model import SlipModel
from ..slip.slip_model import X, X_DOT, X_DDOT, Z, Z_DOT, Z_DDOT

g = SlipModel.g
STANCE_SIM_POINTS = 30


class SlipTargetStateGenerator:

    def __init__(self, slip_model: SlipModel, target_state_weights: ndarray):
        assert target_state_weights.shape == (6,)
        self._slip_model = slip_model
        self._target_state_weights = target_state_weights

    def get_target_cartesian_state(self, td_state, prev_to_state, stance_time, **kwargs) -> ndarray:
        return self.get_symmetrical_to_state(td_state, prev_to_state, stance_time)

    def get_target_state_weights(self, **kwargs):
        return self._target_state_weights

    def get_symmetrical_to_state(self, td_state, prev_to_state, stance_time):
        """
        Default method for generating symmetrical stance trajectories, that conserve linear momentum at TD and TO.
        """
        x_td = float(td_state[X])
        to_des_state = np.zeros((6,))
        to_des_state[X] = prev_to_state[X_DOT] * stance_time + x_td
        to_des_state[X_DOT] = prev_to_state[X_DOT]
        to_des_state[X_DDOT] = 0.0
        to_des_state[Z] = prev_to_state[Z]
        to_des_state[Z_DOT] = -td_state[Z_DOT]
        to_des_state[Z_DDOT] = -g
        return to_des_state


class ForwardSpeedStateGenerator(SlipTargetStateGenerator):

    def __init__(self, slip_model, target_state_weights, desired_forward_speed, desired_duty_cycle):
        super().__init__(slip_model, target_state_weights)
        if isinstance(desired_forward_speed, float):
            self._des_forward_speed = lambda x: desired_forward_speed
        else:
            self._des_forward_speed = desired_forward_speed
        if isinstance(desired_duty_cycle, float):
            self._des_duty_cycle = lambda x: desired_duty_cycle
        else:
            self._des_duty_cycle = desired_duty_cycle

    def get_target_cartesian_state(self, td_state, prev_to_state, stance_time, **kwargs) -> ndarray:
        cycle_in_future = kwargs['cycle_in_future']
        if cycle_in_future:
            to_state = self.get_symmetrical_to_state(td_state, prev_to_state, stance_time)
        else:
            to_state = (self.get_to_state_control_setpoint)(td_state, prev_to_state, stance_time, **kwargs)
        return to_state

    def get_target_state_weights(self, **kwargs):
        cycle_in_future = kwargs['cycle_in_future']
        if cycle_in_future:
            return np.array([0.0, 1.0, 0.0, 1.0, 1.0, 0.0])
        else:
            return self._target_state_weights

    def get_to_state_control_setpoint(self, td_state, prev_to_state, stance_time, **kwargs):
        t = kwargs['t_to']
        x_dot_desired = self._des_forward_speed(t)
        duty_cycle = self._des_duty_cycle(t)
        assert 1.0 >= duty_cycle >= 0.2, 'Duty cycle (Time stance/Time of Cycle) must be in the range [0.2, 1.0]'
        x_td = td_state[X]
        expected_flight_duration = stance_time * (1 - duty_cycle) / duty_cycle
        to_des_state = np.zeros((6,))
        to_des_state[X] = np.mean([x_dot_desired, prev_to_state[X_DOT]]) * stance_time + x_td
        to_des_state[X_DOT] = x_dot_desired
        to_des_state[X_DDOT] = 0.0
        to_des_state[Z] = self._slip_model.r0
        to_des_state[Z_DOT] = expected_flight_duration * g / 2.0
        to_des_state[Z_DDOT] = -g
        return to_des_state


class CycleStateGenerator(SlipTargetStateGenerator):

    def __init__(self, slip_model, target_state_weights, cycle_to_states, cycle_to_weights_mask):
        super().__init__(slip_model, target_state_weights)
        if not len(cycle_to_states) == len(cycle_to_weights_mask):
            raise AssertionError
        elif not len(cycle_to_states):
            raise AssertionError
        self._des_to_states = cycle_to_states
        self._target_state_weights_mask = cycle_to_weights_mask

    def get_target_cartesian_state(self, td_state, prev_to_state, stance_time, **kwargs) -> ndarray:
        cycle = kwargs['cycle'] - 1
        n = cycle % len(self._des_to_states)
        w = self._des_to_states[n]
        return self._des_to_states[n]

    def get_target_state_weights(self, **kwargs):
        cycle = kwargs['cycle'] - 1
        n = cycle % len(self._des_to_states)
        w = self._target_state_weights_mask[n]
        return self._target_state_weights * self._target_state_weights_mask[n]
