
import copy
import sys
from typing import Optional, Dict

import numpy as np
from tqdm import tqdm

from .slip_gait_cycle import SlipGaitCycleCtrl
from .slip_trajectory import SlipTrajectory


class TrajectoryTreeNode:

    def __init__(self, gait_cycle: Optional[SlipGaitCycleCtrl], child_nodes: Dict[(float, 'TrajectoryTreeNode')],
                 parent_node: Optional['TrajectoryTreeNode'], is_root=False):
        self.gait_cycle = gait_cycle
        self.parent_node = parent_node
        self.child_nodes = child_nodes
        self.is_root = is_root

        self._init_to_state = None
        self._optimal_future_node = None
        self._optimal_td_angle = None
        self._frozen = False

    def compute_opt_cost(self):
        present_cycle_cost = self.gait_cycle.optimization_cost
        if len(self.child_nodes) == 0:
            return present_cycle_cost
        else:
            return present_cycle_cost + self.optimal_child_node.compute_opt_cost()

    def freeze(self):
        self._optimal_td_angle = self.optimal_td_angle
        self._frozen = True

    def get_optimal_trajectory(self) -> SlipTrajectory:
        print('Combining optimal cycles into a single trajectory')
        present_node = self if not self.is_root else self.optimal_child_node
        opt_traj = SlipTrajectory(slip_model=(present_node.gait_cycle.slip_model), slip_gait_cycles=[
            copy.deepcopy(present_node.gait_cycle)])
        present_node = present_node.optimal_child_node
        while present_node is not None and present_node.gait_cycle is not None:
            opt_traj.append(present_node.gait_cycle)
            present_node = present_node.optimal_child_node

        return opt_traj

    @property
    def gait_cycle_optimization_cost(self) -> float:
        if self.gait_cycle is not None:
            return self.gait_cycle.optimization_cost
        else:
            return np.NaN

    @property
    def optimal_child_node(self) -> Optional['TrajectoryTreeNode']:
        if len(self.child_nodes) == 0 or self.optimal_td_angle is None:
            return
        else:
            return self.child_nodes[self.optimal_td_angle]

    @property
    def optimal_td_angle(self) -> Optional[float]:
        if len(self.child_nodes) == 0:
            return
        if self._frozen:
            return self._optimal_td_angle

        td_angles = list(self.child_nodes.keys())
        costs = [node.compute_opt_cost() for node in self.child_nodes.values()]
        if np.all(np.isnan(costs)):
            self._optimal_td_angle = td_angles[(-1)]
        else:
            best_branch_idx = np.nanargmin(costs)
        self._optimal_td_angle = td_angles[best_branch_idx]
        return self._optimal_td_angle

    @property
    def final_to_state(self):
        if self.is_root:
            return self._init_to_state
        else:
            return self.gait_cycle.take_off_state

    @property
    def final_to_angle(self):
        if self.is_root:
            return np.pi / 2
        else:
            return self.gait_cycle.take_off_angle

    @property
    def cycle_number(self):
        if self.is_root:
            return 0
        else:
            return self.parent_node.cycle_number + 1

    @property
    def node_height(self):
        if len(self.child_nodes) == 0:
            return 0
        else:
            return max([node.node_height for node in self.child_nodes.values()])

    @property
    def n_branches(self):
        return len(self.child_nodes)

    @property
    def tree_size(self):
        return self.n_branches + sum([node.tree_size for node in self.child_nodes.values()])

    def __str__(self):
        return 'n:%d_cost:%.3f_children:%d_future_branches:%d' % (
            self.cycle_number, self.gait_cycle_optimization_cost, self.n_branches, self.tree_size)

    def __repr__(self):
        return 'n:%d_cost:%.3f_children:%d_future_branches:%d' % (
            self.cycle_number, self.gait_cycle_optimization_cost, self.n_branches, self.tree_size)

    def plot(self, optimal_color=(0, 0, 0, 1), sub_optimal_color=(1, 0, 0, 0.15), unfeasible_color=(0, 0, 1, 0.15)):
        from slip_control.utils.plot_utils import plot_slip_trajectory
        present_node = self
        n_cycles = present_node.node_height + 1
        outer_pbar = tqdm(total=n_cycles, desc='Trajectory tree plot | Gait Cycles', position=0, leave=True,
                          file=(sys.stdout))
        plt_axs = None
        while len(present_node.child_nodes) > 0:
            cycle_start_time = present_node.gait_cycle.start_time if not present_node.is_root else 0.0
            optimal_node = present_node.optimal_child_node
            plt_axs = plot_slip_trajectory((optimal_node.gait_cycle), color=optimal_color, plot_passive=True,
                                           plt_axs=plt_axs,
                                           plot_td_angles=True)
            for theta_td, child_node in present_node.child_nodes.items():
                if theta_td == present_node.optimal_td_angle:
                    continue
                plot_passive = np.isinf(child_node.gait_cycle_optimization_cost)
                color = unfeasible_color if plot_passive else sub_optimal_color
                plt_axs = plot_slip_trajectory((child_node.gait_cycle), color=color,
                                               plt_axs=plt_axs,
                                               plot_passive=plot_passive,
                                               plot_td_angles=False)

            present_node = optimal_node
            outer_pbar.update()

        outer_pbar.close()
        return plt_axs.get_figure()
