from math import pi as PI

import numpy as np
from scipy.integrate import solve_ivp

X, X_DOT, X_DDOT, Z, Z_DOT, Z_DDOT = (0, 1, 2, 3, 4, 5)
THETA, THETA_DOT, R, R_DOT = (0, 1, 2, 3)
MIN_TD_ANGLE = np.deg2rad(35)
MAX_TD_ANGLE = np.deg2rad(145)


# noinspection PyTypeChecker
class SlipModel:
    g = 9.81

    def __init__(self, mass, leg_length, k_rel, verbose=False):
        self.m = mass
        self.r0 = leg_length
        self._k_rel = k_rel
        self.k = self._k_rel * self.m * SlipModel.g / self.r0
        self.verbose = verbose
        if verbose:
            print(str(self))

    def get_flight_trajectory(self, t, take_off_state):
        assert take_off_state.shape[0] == 6, 'Provide a valid (6,) cartesian take-off state'
        x_TO, x_dot_TO, x_ddot_TO, z_TO, z_dot_TO, z_ddot_TO = take_off_state

        flight_traj = np.zeros((6, t.shape[0]))
        flight_traj[0, :] = x_TO + x_dot_TO * t
        flight_traj[1, :] += x_dot_TO
        flight_traj[2, :] += 0.0
        flight_traj[3, :] = z_TO + z_dot_TO * t - 0.5 * SlipModel.g * t ** 2
        flight_traj[4, :] = z_dot_TO - SlipModel.g * t
        flight_traj[5, :] += -SlipModel.g

        return flight_traj

    def get_stance_trajectory(self, touch_down_state_polar, dt=0.005):
        """
        Function to obtain the passive trajectory of the SLIP model given a touch-down state. This function uses 
        numerical integration of RK-4th order to integrate the dynamics of the SLIP model until a take-off event is 
        detected or the model mass makes contact with the fround i.e. theta= 180 or 0 [deg]
        :param touch_down_state_polar: (4,) Touch down state in polar coordinates 
        :param dt: Integration time-step
        :return: t: (k,) Time signal of the integration method, of size `k` where `k` is the iteration of integration 
                         termination.
                 stance_traj: (4, k) Stance trajectory of the SLIP passive model in polar coordinates. 
        """
        assert touch_down_state_polar.shape[0] == 4, 'Provide a valid (4,) polar touch-down state'

        def slip_stance_dynamics(t, x, m, r0, g, k):
            theta, theta_dot, r, r_dot = x
            x_dot = np.zeros((4,))
            x_dot[0] = theta_dot
            x_dot[1] = -2 * r_dot * theta_dot * (1 / r) - g * (1 / r) * np.cos(theta)
            x_dot[2] = r_dot
            x_dot[3] = -g * np.sin(theta) + theta_dot ** 2 * r + k / m * (r0 - r)
            return x_dot

        def take_off_detection(t, x, *args):
            return x[2] - self.r0

        to_event = take_off_detection
        to_event.terminal = True
        to_event.direction = 1

        def fall_detection(t, x, *args):
            return np.sin(x[0])

        fall_event = fall_detection
        fall_event.terminal = True
        fall_event.direction = -1
        solution = solve_ivp(fun=slip_stance_dynamics, t_span=(0, 2 / self.spring_natural_freq),
                             t_eval=(np.linspace(0, 1/self.spring_natural_freq, int(1/self.spring_natural_freq/dt))),
                             y0=touch_down_state_polar,
                             args=(self.m, self.r0, SlipModel.g, self.k),
                             events=[to_event, fall_event],
                             first_step=0.0001)
        t = solution.t
        stance_traj = solution.y
        if solution.status == 1:
            try:
                # Include TO event state into time and trajectory
                t = np.append(t, solution.t_events[0])
                stance_traj = np.hstack((stance_traj, solution.y_events[0].T))
            except:
                pass

        return t, stance_traj

    def cartesian_to_polar(self, trajectory, foot_contact_pos: float = 0.0):
        """
        Utility function to convert an SLIP CoM trajectory in cartesian coordinates to polar.
        This function assumes the input trajectory is actuated and therefore calculates the hip torque and leg
        length offset required to achieve the input trajectory. This control inputs refer to the extended SLIP model
        presented in "Optimal Control of a Differentially Flat Two-Dimensional Spring-Loaded Inverted Pendulum Model".
        See "Learning to run naturally: guiding policies with the Spring-Loaded Inverted Pendulum" Chap 3.1 & 4.1 for
        more details.
        :param trajectory: (6, k) Cartesian trajectory [x, xdot, xddot, z, zdot, zzdot] through time of a SLIP model
                            during stance phase.
        :param foot_contact_pos:  Foot contact X coordinate. This will become the reference frame position in
                                  polar coordinates
        :return: tuple(polar_trajectory, control_input).
            polar_trajectory: (4, k) Polar trajectory [theta, theta_dot, r, rdot] of the SLIP model during
            stance phase.
            control_input: (2, k) Resting leg length displacement and hip torque required to achieve the input
            cartesian trajectory. If the input trajectory is passive the control inputs become zero vectors.

        """
        cart_traj = np.array(trajectory)
        assert cart_traj.shape[0] == 6, 'Provide a valid (6, k) cartesian trajectory'
        if len(cart_traj.shape) == 1:
            cart_traj = cart_traj[:, None]

        g = 9.81
        x = cart_traj[0, :]
        x_dot = cart_traj[1, :]
        x_ddot = cart_traj[2, :]
        z = cart_traj[3, :]
        z_dot = cart_traj[4, :]
        z_ddot = cart_traj[5, :]

        # Center the x dimension according to desired reference frame
        x -= foot_contact_pos

        epsilon = 1e-06
        y12_squared = x ** 2 + z ** 2 + epsilon

        theta = (np.arctan2(z, x) + 2 * PI) % (2 * PI)
        theta_dot = (x * z_dot - z * x_dot) / y12_squared
        r = np.sqrt(y12_squared)
        r_dot = (x * x_dot + z * z_dot) / r

        leg_length_shift = r + self.m * (x * x_ddot + z * z_ddot + g * z) / (self.k * r) - self.r0
        hip_torque = self.m * (g * x + x * z_ddot - z * x_ddot)

        polar_traj = np.stack((theta, theta_dot, r, r_dot), axis=0)
        control_input = np.stack((leg_length_shift, hip_torque), axis=0)

        return (np.squeeze(polar_traj), np.squeeze(control_input))

    def polar_to_cartesian(self, trajectory, control_signal=None, foot_contact_pos=0.0):
        """
        Utility function to convert an actuated SLIP CoM trajectory in polar coordinates to cartesian.
        This function assumes the extended SLIP model presented in "Optimal Control of a Differentially Flat
        Two-Dimensional Spring-Loaded Inverted Pendulum Model".
        :param trajectory: (4, k) Polar trajectory of the SLIP model during stance phase
        :param control_signal: (2, k) Optional leg length displacement and hip torque of exerted at every timestep during the
        stance phase.
        :param foot_contact_pos: Cartesian x coordinate of the foot contact point during the stance phase of the input
        trajectory.
        :return:
        """
        polar_traj = np.array(trajectory)
        assert polar_traj.shape[0] == 4, 'Provide a valid (4, k) polar trajectory: %s' % polar_traj.shape
        if polar_traj.ndim == 1:
            polar_traj = np.expand_dims(polar_traj, axis=1)
            u_ctrl = control_signal
        else:
            if control_signal is None:
                u_ctrl = np.zeros((2, polar_traj.shape[(-1)]))
            else:
                u_ctrl = np.array(control_signal)
            if u_ctrl.ndim == 1:
                u_ctrl = np.expand_dims(u_ctrl, axis=1)
            assert u_ctrl.shape[0] == 2, 'Provide a valid (2, k) control input vector'
            assert polar_traj.shape[1] == u_ctrl.shape[1], 'Len of trajectory: polar = %d | control_input = %d' % (
                polar_traj.shape[1], u_ctrl.shape[1])
        theta = polar_traj[0, :]
        theta_dot = polar_traj[1, :]
        r = polar_traj[2, :]
        r_dot = polar_traj[3, :]

        r_delta = u_ctrl[0, :]
        tau_hip = u_ctrl[1, :]

        x = np.cos(theta) * r
        x_dot = -np.sin(theta) * theta_dot * r + np.cos(theta) * r_dot
        x_ddot = self.k/self.m * np.cos(theta) * (self.r0 - r + r_delta) - np.sin(theta) / (self.m * r) * tau_hip
        z = np.sin(theta) * r
        z_dot = np.cos(theta) * theta_dot * r + np.sin(theta) * r_dot
        z_ddot = self.k/self.m * np.sin(theta) * (self.r0 - r + r_delta) + np.cos(theta) / (self.m*r) * tau_hip - self.g

        # Center x dimension
        x += foot_contact_pos

        cartesian_traj = np.stack((x, x_dot, x_ddot, z, z_dot, z_ddot), axis=0)
        return cartesian_traj if len(trajectory.shape) == 2 else np.squeeze(cartesian_traj)

    def predict_td_state(self, TO_init_state, theta_TD):
        """
        Function to analyze the ballistic trajectory imposed by the Take Off (TO) conditions and to predict the most optimal
        Touch Down (TD) ref_state ([y, x_dot_des, x_ddot, z, z_dot, z_ddot]) when a desired angle angle (`theta_td`) is desired at
        Touch Down.
        The function handles 3 cases:
        1). Whenever the desired TD angle can be reach at touch down during the ballistic trajectory WITHOUT leg
        pre-compression. i.e. The TD event will occur with the leg at its nominal length `r0`.
        2). Whenever the desired TD angle CANNOT be reach at touch down during the ballistic trajectory WITHOUT leg
        pre-compression, but the initial vertical velocity is positive and some ballistic motion is expected. In this case,
        the touch down event will occur at the apex of the flight trajectory minimizing the leg pre-compression required.
        3). Whenever the desired TD angle CANNOT be reach at touch down during the ballistic trajectory WITHOUT leg
        pre-compression, and the initial vertical velocity is negative. In this case, the touch down event will occur at in
        the shortest time possible. i.e. If the desired TD angle requires a vertical position greater than the current one
        TD will occur instantly. Otherwise case 1) will apply.

        :param TO_init_state: Cartesian state ([y, x_dot_des, x_ddot, z, z_dot, z_ddot]) indicating the initial conditions of
        the ballistic trajectory
        :param theta_TD: Desired Touch Down angle measured from the positive `y` horizontal axis to the COM of the SLIP body
        measured counter-clockwise
        :param r0: Nominal leg length of the SLIP model
        :return: [TD_state_cartesian, time_of_flight, x_foot_TD_pos]
            TD_state_cartesian: Cartesian state at touch down [y, x_dot_des, x_ddot, z, z_dot, z_ddot].
            time_of_flight: The time in seconds needed to go from Take Off to Touch Down
            x_foot_TD_pos: Horizontal position of contact point between the foot and the ground
        """
        x_init, x_dot_init, x_ddot_init, z_init, z_dot_init, z_ddot_init = TO_init_state
        td_state_cartesian = np.zeros((6,))
        time_of_flight = self.get_flight_time(TO_init_state, theta_TD)
        foot_contact_pos = None

        # Case #1: Desired TD angle will be reached through ballistic motion
        # i.e. z_apex > r0*sin(theta_td)
        if time_of_flight > 0:
            td_state_cartesian[0] = x_init + x_dot_init * time_of_flight
            td_state_cartesian[1] = x_dot_init
            td_state_cartesian[2] = x_ddot_init
            td_state_cartesian[3] = self.r0 * np.sin(theta_TD)
            td_state_cartesian[4] = -np.sqrt(z_dot_init ** 2 - 2 * SlipModel.g * (self.r0 * np.sin(theta_TD) - z_init))
            td_state_cartesian[5] = -SlipModel.g
            foot_contact_pos = td_state_cartesian[0] - self.r0 * np.cos(theta_TD)
        # Case #2: Desired TD angle will not be reached through ballistic motion
        # However z_dot_init > 0, i.e. there is some ballistic motion that can be
        # exploited to gain zome height before Touch Down.
        # i.e. z_apex < r0*sin(theta_td)
        elif time_of_flight <= 0 and z_dot_init > 0.0:
            z_apex = z_init + z_dot_init ** 2 / (2 * SlipModel.g)
            t_apex = z_dot_init / SlipModel.g

            td_state_cartesian[0] = x_init + x_dot_init * t_apex
            td_state_cartesian[1] = x_dot_init
            td_state_cartesian[2] = x_ddot_init
            td_state_cartesian[3] = z_apex
            td_state_cartesian[4] = 0
            td_state_cartesian[5] = -SlipModel.g

            foot_contact_pos = td_state_cartesian[0] - td_state_cartesian[3] * np.cos(theta_TD)
            time_of_flight = t_apex
        # Case #3: There will be no height increase with the ballistic trajectory,
        # and the model should try to recover immediately by imposing touch down in
        # the act.
        elif time_of_flight <= 0 and z_dot_init <= 0.0:
            td_state_cartesian[0] = x_init + x_dot_init * 0.0
            td_state_cartesian[1] = x_dot_init
            td_state_cartesian[2] = x_ddot_init
            td_state_cartesian[3] = z_init
            td_state_cartesian[4] = z_dot_init
            td_state_cartesian[5] = z_ddot_init
            foot_contact_pos = td_state_cartesian[0] - td_state_cartesian[3] * np.cos(theta_TD)
        return td_state_cartesian, time_of_flight, foot_contact_pos

    def get_flight_time(self, take_off_init_state, touch_down_angle: float):
        """
        Function to calculate the time of flight from a take-off state until a touch-down event with leg angle
        at `touch_down_angle` occurs.
        :param take_off_init_state: (6,) Cartesian initial take-off state
        :param touch_down_angle: Angle of SLIP leg at the touch-down event. Counterclockwise from the horizontal axis
        :return: Expected time of flight of the SLIP model
        """
        _, _, _, z_init, z_dot_init, _ = take_off_init_state
        if z_dot_init ** 2 > 2 * SlipModel.g * (self.r0 * np.sin(touch_down_angle) - z_init):
            time_of_flight = 1 / SlipModel.g * (z_dot_init + np.sqrt(
                z_dot_init ** 2 - 2 * SlipModel.g * (self.r0 * np.sin(touch_down_angle) - z_init)))
        else:
            time_of_flight = np.array(0.0)
        if np.isinf(time_of_flight):
            raise Exception
        return time_of_flight

    @property
    def k_rel(self):
        return self._k_rel

    @property
    def spring_natural_freq(self):
        return 1 / (2 * PI) * np.sqrt(self.k / self.m)

    @staticmethod
    def get_take_off_from_apex(des_apex_state):
        x_to, xdot_to, xddot_to, z_to, zdot_to, zddot_to = des_apex_state
        t_apex = zdot_to / SlipModel.g
        apex_state = np.array([x_to + xdot_to * t_apex,
                               xdot_to,
                               0,
                               z_to + zddot_to ** 2 / (2 * SlipModel.g),
                               0.0,
                               -SlipModel.g])
        return apex_state

    @staticmethod
    def get_apex_from_take_off(to_state):
        x_to, xdot_to, xddot_to, z_to, zdot_to, zddot_to = to_state
        t_apex = zdot_to / SlipModel.g
        return np.array([x_to + xdot_to * t_apex,
                         xdot_to,
                         0,
                         z_to + zddot_to ** 2 / (2 * SlipModel.g),
                         0.0,
                         -SlipModel.g])

    def __str__(self):
        return 'SLIP m=%.2f[Kg] r0= %.1f[m] k=%.2f[N/m] k_rel=%.1f[.] ' % (self.m, self.r0, self.k, self._k_rel)

    def __repr__(self):
        return str(self)
