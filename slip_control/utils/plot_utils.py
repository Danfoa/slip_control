# uncompyle6 version 3.7.4
# Python bytecode 3.6 (3379)
# Decompiled from: Python 3.6.13 |Anaconda, Inc.| (default, Feb 23 2021, 21:15:04) 
# [GCC 7.3.0]
# Embedded file name: /home/dordonez/Projects/slip_control/slip_control/utils/plot_utils.py
# Compiled at: 2021-09-30 19:22:48
# Size of source mod 2**32: 22201 bytes
from typing import Optional, Union, List
import numpy as np, matplotlib.pyplot as plt
from ..slip.slip_trajectory import SlipTrajectory
from ..slip.slip_gait_cycle import SlipGaitCycleCtrl, SlipGaitCycle
from ..slip.slip_model import X_DOT, X_DDOT, Z_DOT, Z_DDOT, THETA, THETA_DOT, R, R_DOT

LABEL_FONT_SIZE = 15

def plot_cartesian_trajectory(traj, t, target_state=None, foot_pos=None, touch_down_angle=None, plt_axs=None, color='k',
                              label=None, marker_size=2, linestyle='-', marker='.', area_color=None, put_legend=True,
                              fig_size=(10, 10)):
    if not traj.shape[0] == 6:
        raise AssertionError("Expected cartesian 2D trajectory [x,x',x'',z,z',z'']")
    else:
        if not traj.shape[(-1)] == t.size:
            raise AssertionError('`t` %s and `traj` %s must have same lengths' % (t.shape, traj.shape))
        else:
            if plt_axs is None:
                fig, axs = plt.subplots(4, 2, figsize=fig_size)
                yz_gs = axs[(0, 0)].get_gridspec()
                for ax in axs[0, :]:
                    ax.remove()

                xz_ax = fig.add_subplot(yz_gs[0, :])
                x_axs = axs[1:, 0]
                z_axs = axs[1:, 1]
                plt_axs = [xz_ax, x_axs, z_axs]
            else:
                fig = plt_axs[0].get_figure()
                xz_ax, x_axs, z_axs = plt_axs[0], plt_axs[1], plt_axs[2]
            if target_state is None:
                target_state = np.empty(6)
                target_state[:] = np.nan
            assert target_state.shape == (6,)
            xz_ax.plot((traj[0, :]), (traj[3, :]), '-', label=label, color=color, markersize=marker_size,
                       linestyle=linestyle, marker=marker)
            if area_color is not None:
                xz_ax.fill_between((traj[0, :]), (traj[3, :]), alpha=0.1, color=area_color)
            if foot_pos is not None:
                xz_ax.plot(foot_pos, 0, color=color, markersize='8', marker='o', markeredgecolor='k')
                xz_ax.plot((traj[(0, 0)]), (traj[(3, 0)]), color=color, marker='o', markersize='6', markeredgecolor='k')
                n = 4
                coarse_idx = np.round(np.linspace(0, traj.shape[1] - 1, n)).astype(int)
                yz_coarse_traj = traj[:, coarse_idx]
                for i in range(n):
                    xz_ax.plot([foot_pos, yz_coarse_traj[(0, i)]], [0, yz_coarse_traj[(3, i)]], color=color, alpha=0.15,
                               linestyle=linestyle,
                               marker='o',
                               markersize=3)

            if touch_down_angle is not None:
                xz_ax.text(foot_pos + 0.01, 0.05, '%.1f°' % touch_down_angle)
        xz_ax.grid('on', alpha=0.15)
        xz_ax.set_ylim(bottom=0.0, auto=True)
        xz_ax.set_xlabel('$\\mathbf{y[m]}$', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        xz_ax.set_ylabel('$\\mathbf{z[m]}$', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        if put_legend:
            xz_ax.legend(bbox_to_anchor=(1.0, 1), loc='upper left')
    x_titles = ['$\\mathbf{x [m]}$', '$\\mathbf{\\dot{x} [m/s]}$', '$\\mathbf{\\ddot{x} [m/s^2]}$']
    for i, ax in enumerate(x_axs):
        ax.plot(t, (traj[i, :]), 'k-o', label=label, color=color, markersize=marker_size, linestyle=linestyle,
                marker=marker)
        ax.plot((t[(-1)]), (target_state[i]), 'D', color=color, markersize='6', markeredgecolor='k')
        ax.set_ylabel((x_titles[i]), fontweight='bold', fontsize=LABEL_FONT_SIZE)
        if i == len(x_axs) - 1:
            ax.set_xlabel('t[s]', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.grid('on', alpha=0.15)
        ax.set_xlim(xmin=0.0, auto=True)

    z_titles = ['$\\mathbf{z [m]}$', '$\\mathbf{\\dot{z} [m/s]}$', '$\\mathbf{\\ddot{z} [m/s^2]}$']
    for i, ax in enumerate(z_axs):
        ax.plot(t, (traj[i + 3, :]), 'k-o', label=label, color=color, markersize=marker_size, linestyle=linestyle,
                marker=marker)
        ax.plot((t[(-1)]), (target_state[(i + 3)]), 'D', color=color, markersize='6', markeredgecolor='k')
        ax.set_ylabel((z_titles[i]), fontweight='bold', fontsize=LABEL_FONT_SIZE)
        if i == len(z_axs) - 1:
            ax.set_xlabel('t[s]', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.grid('on', alpha=0.15)
        ax.set_xlim(xmin=0.0, auto=True)

    return (fig, plt_axs)


def plot_polar_trajectory(polar_traj, t, target_state=None, axs=None, color='k', label='', target_color='b'):
    if axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))
        r_axs = axs[:, 1]
        theta_axs = axs[:, 0]
    else:
        fig = None
        r_axs = axs[:, 1]
        theta_axs = axs[:, 0]
    if target_state is None:
        target_state = np.empty(4)
        target_state[:] = np.nan
    elif not target_state.shape == (4,):
        raise AssertionError
    y_titles = [
        '$r$', '$\\dot{r}$', '$\\ddot{r}$']
    for i, ax in enumerate(r_axs.flatten()):
        ax.plot(t, (polar_traj[i + 2, :]), 'k-', markersize='2', label=label, color=color)
        ax.plot((t[0]), (polar_traj[(i + 2, 0)]), color=color, marker='o', markersize='6', markeredgecolor='k')
        ax.plot((t[(-1)]), (target_state[(i + 2)]), 'D', color=target_color, markersize='10')
        ax.set_ylabel((y_titles[i]), rotation=0)
        ax.grid('on', alpha=0.15)

    z_titles = ['$\\theta$', '$\\dot{\\theta}$', '$\\ddot{\\theta}$']
    for i, ax in enumerate(theta_axs.flatten()):
        ax.plot(t, (np.rad2deg(polar_traj[i, :])), 'k-', markersize='2', label=label, color=color)
        ax.plot((t[0]), (np.rad2deg(polar_traj[(i, 0)])), color=color, marker='o', markersize='6', markeredgecolor='k')
        ax.plot((t[(-1)]), (np.rad2deg(target_state[i])), 'D', color=target_color, markersize='10')
        ax.set_ylabel((z_titles[i]), rotation=0)
        ax.grid('on', alpha=0.15)

    return (
        fig, axs)


def plot_control_trajectory(u_traj, t, theta, r, r0, k, axs=None, color='k', label='',
                            limit_color='-', marker_size=2, linestyle='-', marker='.', fig_size=(15, 10)):
    if not u_traj.shape[0] == 2:
        raise AssertionError
    else:
        if axs is None:
            fig, axs = plt.subplots(2, 2, figsize=fig_size)
        else:
            fig = None
    displacement = u_traj[0, :]
    hip_torque = u_traj[1, :]
    extension_ax = axs[:, 0]
    torque_ax = axs[:, 1]
    ax = extension_ax[0]
    ax.set_title('Leg length displacement $u_{1}$')
    ax.plot(t, displacement, 'k-o', label=label, color=color, markersize=marker_size, linestyle=linestyle,
            marker=marker)
    ax.set_ylabel('Displacement [m]')
    ax.grid('on', alpha=0.15)
    ax = extension_ax[1]
    ax.set_title('Leg length displacement $u_{1}$')
    resultant_force = k * (r0 - r + u_traj[0, :])
    passive_force = k * (r0 - r)
    ax.plot((theta * 180 / np.pi), resultant_force, 'k-o', label=(label + '[total]'), color=color,
            markersize=marker_size, linestyle='--',
            marker=marker)
    ax.plot((theta * 180 / np.pi), passive_force, 'k-o', label=(label + '[disp]'), color=color,
            markersize=(marker_size / 2), linestyle=linestyle,
            marker=marker)
    ax.fill_between((theta * 180 / np.pi), resultant_force, passive_force, alpha=0.1, color=color)
    ax.set_ylabel('Radial Force [m]')
    ax.set_xlabel('Hip angle $\\theta$ [Deg]')
    ax.grid('on', alpha=0.15)
    ax = torque_ax[0]
    ax.set_title('Hip torque $u_{2}$')
    ax.plot(t, hip_torque, 'k-o', label=label, color=color, markersize=marker_size, linestyle=linestyle, marker=marker)
    ax.set_ylabel('Torque [Nm]')
    ax.set_xlabel('Time [s]')
    ax.grid('on', alpha=0.15)
    ax = torque_ax[1]
    ax.set_title('Hip torque $u_{2}$ vs $\\theta$')
    ax.plot((theta * 180 / np.pi), hip_torque, 'k-o', label=label, color=color, markersize=marker_size,
            linestyle=linestyle,
            marker=marker)
    ax.fill_between((theta * 180 / np.pi), hip_torque, alpha=0.1, color=color)
    ax.set_ylabel('Torque [Nm]')
    ax.set_xlabel('Hip angle $\\theta$ [Deg]')
    ax.grid('on', alpha=0.15)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    return (
        fig, axs)


def plot_slip_trajectory(slip_traj: Union[(SlipTrajectory, SlipGaitCycleCtrl)], color='k', plot_passive=True,
                         plt_axs=None, title=None, plot_td_angles=False):
    if isinstance(slip_traj, SlipGaitCycle):
        cycles = [
            slip_traj]
    else:
        cycles = slip_traj.gait_cycles
    for gait_cycle in cycles:
        touch_down_angle = np.rad2deg(gait_cycle.touch_down_angle) if plot_td_angles else None
        # Plot flight trajectory
        fig, plt_axs = plot_cartesian_trajectory(traj=(gait_cycle.flight_cartesian_traj), t=(gait_cycle.t_flight),
                                                 color=color,
                                                 plt_axs=plt_axs,
                                                 linestyle='dotted',
                                                 area_color=(0.215, 0.317, 0.403),
                                                 marker_size=1,
                                                 put_legend=False,
                                                 fig_size=(15, 15))
        # Plot passive trajectory if available
        if plot_passive and isinstance(gait_cycle, SlipGaitCycleCtrl) and gait_cycle.passive_gait_cycle is not None:
            passive_cycle = gait_cycle.passive_gait_cycle
            fig, plt_axs = plot_cartesian_trajectory(traj=(passive_cycle.stance_cartesian_traj),
                                                     t=(passive_cycle.t_stance),
                                                     color=color,
                                                     plt_axs=plt_axs,
                                                     linestyle='--',
                                                     marker='None',
                                                     put_legend=False,
                                                     foot_pos=(gait_cycle.foot_contact_pos),
                                                     marker_size=3)
        # Plot stance trajectory
        fig, plt_axs = plot_cartesian_trajectory(traj=(gait_cycle.stance_cartesian_traj), t=(gait_cycle.t_stance),
                                                 color=color,
                                                 plt_axs=plt_axs,
                                                 linestyle='-',
                                                 area_color=(1, 0.647, 0.16),
                                                 marker='',
                                                 put_legend=False,
                                                 foot_pos=(gait_cycle.foot_contact_pos),
                                                 target_state=(gait_cycle.target_to_state),
                                                 touch_down_angle=touch_down_angle)

    fig = plt_axs[0].get_figure()
    subtitle = str(slip_traj.slip_model)
    if title is None:
        title = subtitle
    else:
        title += '\n' + subtitle
    fig.suptitle(title, weight='demibold')
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    return plt_axs


def plot_limit_cycles(slip_traj: SlipTrajectory, axs=None, cmap_name='copper', fig_size=(15, 10)):
    LABEL_FONT_SIZE = 12
    cmap = plt.cm.get_cmap(cmap_name)

    def plot_limit_cycle_metric(x_flight, y_flight, x_stance, y_stance, x_name, y_name, flight_phase, stance_phase, ax,
                                ax2, ax3, color):
        if x_flight is not None:
            ax.plot(x_flight, y_flight, label=label, color=color, markersize=marker_size, linestyle='dotted')
        ax.plot(x_stance, y_stance, label=label, color=color, markersize=marker_size, linestyle='-')
        ax.plot((x_stance[0]), (y_stance[0]), 'D', color=color, markersize='4', markeredgecolor='k')
        ax.plot((x_stance[(-1)]), (y_stance[(-1)]), 'X', color=color, markersize='4', markeredgecolor='k')
        ax.set_xlabel(x_name, fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.set_ylabel(y_name, fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax.grid('on', alpha=0.15)
        if x_flight is not None:
            ax2.plot(flight_phase, y_flight, label=label, color=color, markersize=marker_size, linestyle='dotted')
        ax2.plot(stance_phase, y_stance, label=label, color=color, markersize=marker_size, linestyle='-')
        ax2.plot((stance_phase[0]), (y_stance[0]), 'D', color=color, markersize='4', markeredgecolor='k')
        ax2.plot((stance_phase[(-1)]), (y_stance[(-1)]), 'X', color=color, markersize='4', markeredgecolor='k')
        ax2.set_ylabel(y_name, fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax2.set_xlabel('$\\phi$', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax2.grid('on', alpha=0.15)
        if x_flight is not None:
            ax3.plot(flight_phase, x_flight, label=label, color=color, markersize=marker_size, linestyle='dotted')
        ax3.plot(stance_phase, x_stance, label=label, color=color, markersize=marker_size, linestyle='-')
        ax3.plot((stance_phase[0]), (x_stance[0]), 'D', color=color, markersize='4', markeredgecolor='k')
        ax3.plot((stance_phase[(-1)]), (x_stance[(-1)]), 'X', color=color, markersize='4', markeredgecolor='k')
        ax3.set_ylabel(x_name, fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax3.set_xlabel('$\\phi$', fontweight='bold', fontsize=LABEL_FONT_SIZE)
        ax3.grid('on', alpha=0.15)

    if axs is None:
        fig, axs = plt.subplots(4, 3, figsize=fig_size)

    for i, cycle in enumerate(slip_traj.gait_cycles):
        color = cmap(i / len(slip_traj))
        marker_size = 1
        label = None
        flight_traj = cycle.flight_cartesian_traj
        stance_traj = cycle.stance_cartesian_traj
        stance_traj_polar = cycle.stance_polar_traj

        flight_phase = slip_traj.get_gait_phase(np.linspace(start=cycle.t_flight[0], stop=cycle.t_flight[-1],
                                                            num=len(cycle.t_flight)))
        stance_phase = slip_traj.get_gait_phase(np.linspace(start=cycle.t_stance[0], stop=cycle.t_stance[-2],
                                                            num=len(cycle.t_stance)))

        plot_limit_cycle_metric(x_flight=(flight_traj[X_DOT]), y_flight=(flight_traj[Z_DOT]),
                                x_stance=(stance_traj[X_DOT]), y_stance=(stance_traj[Z_DOT]),
                                x_name='$\\dot{x}[m/s]}$', y_name='$\\dot{z}[m/s]}$',
                                flight_phase=flight_phase, stance_phase=stance_phase,
                                ax=(axs[(0, 0)]), ax2=(axs[(0, 1)]), ax3=(axs[(0, 2)]), color=color)
        plot_limit_cycle_metric(x_flight=(flight_traj[X_DDOT]), y_flight=(flight_traj[Z_DDOT]),
                                x_stance=(stance_traj[X_DDOT]), y_stance=(stance_traj[Z_DDOT]),
                                x_name='$\\ddot{x}[m/s]}$', y_name='$\\ddot{z}[m/s]}$',
                                flight_phase=flight_phase, stance_phase=stance_phase,
                                ax=(axs[(1, 0)]), ax2=(axs[(1, 1)]), ax3=(axs[(1, 2)]),
                                color=color)
        plot_limit_cycle_metric(x_flight=None, y_flight=None,
                                x_stance=(np.rad2deg(stance_traj_polar[THETA])), y_stance=(stance_traj_polar[R]),
                                x_name='$\\theta[deg]$', y_name='$r[m]$',
                                flight_phase=flight_phase, stance_phase=stance_phase,
                                ax=(axs[(2, 0)]), ax2=(axs[(2, 1)]), ax3=(axs[(2, 2)]),
                                color=color)
        plot_limit_cycle_metric(x_flight=None, y_flight=None,
                                x_stance=(np.rad2deg(stance_traj_polar[THETA_DOT])), y_stance=(stance_traj_polar[R_DOT]),
                                x_name='$\\dot{\\theta}[deg/s]$', y_name='$\\dot{r}[m/s]$',
                                flight_phase=flight_phase, stance_phase=stance_phase,
                                ax=(axs[(3, 0)]), ax2=(axs[(3, 1)]), ax3=(axs[(3, 2)]),
                                color=color)
    plt.tight_layout()
    return axs
