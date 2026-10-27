import os
import torch
import numpy as np
import genesis as gs
from genesis.engine.solvers.rigid.rigid_solver_decomp import RigidSolver

def get_randomization_cfg(
    randomize=False,
    on_friction=False,
    on_com=False,
    on_mass=False,
    external_force=False,
    force_prob=0.2,
    force_scale=10.0,
    torque_scale=1.0,
):
    rand_cfg = dict(randomize=randomize)
    if on_friction:
        rand_cfg['friction'] = (0.5, 1.5)
    if on_com:
        rand_cfg['com'] = (-0.03, 0.03)
    if on_mass: # this is added delta mass
        rand_cfg['mass'] = (-0.1, 0.5)
    rand_cfg['external_force'] = external_force
    rand_cfg['force_scale'] = force_scale
    rand_cfg['torque_scale'] = torque_scale
    rand_cfg['force_prob'] = force_prob
    return rand_cfg

class RandomizationModule:
    """
    Handles randomization of the environment physics
    """
    def __init__(self, rand_cfg, solver, task_object, num_envs):
        self.randomize = rand_cfg.get('randomize', False)
        self.rand_cfg = rand_cfg
        if self.randomize:
            assert isinstance(solver, RigidSolver), "Solver must be RigidSolver"
        self.rigid_solver = solver
        self.num_envs = num_envs
        self.object = task_object

    def on_step(self, episode_length_buf):
        if not self.randomize:
            return
        self._random_force_torque(episode_length_buf)

    def on_reset_idx(self, env_idxs):
        if not self.randomize:
            return
        if self.rand_cfg.get('friction', False):
            self._randomize_link_friction(friction_range=self.rand_cfg['friction'], env_idxs=env_idxs)
        if self.rand_cfg.get('com', False):
            self._randomize_com_displacement(com_range=self.rand_cfg['com'], env_idxs=env_idxs)
        if self.rand_cfg.get('mass', False):
            self._randomize_mass(mass_range=self.rand_cfg['mass'], env_idxs=env_idxs)

    def _randomize_link_friction(self, friction_range=(0.3, 1.5), env_idxs=None):
        solver = self.rigid_solver
        env_idxs = env_idxs if env_idxs is not None else range(self.num_envs)
        _min, _max = friction_range
        ratios = gs.rand((len(env_idxs), 1), dtype=float).repeat(1, solver.n_geoms)
        ratios *= (_max - _min) + _min
        solver.set_geoms_friction_ratio(
            ratios, torch.arange(solver.n_geoms), env_idxs
        )

    def _randomize_com_displacement(self, com_range=(-0.05, 0.05), env_idxs=None):
        if self.object is None:
            return
        solver = self.rigid_solver
        env_idxs = env_idxs if env_idxs is not None else range(self.num_envs)
        _min, _max = com_range
        # randomize object links com
        obj_link_ids = self.object.coll_idxs_global
        n_links = len(obj_link_ids)
        com_displacement = gs.rand((len(env_idxs), n_links, 3), dtype=float)
        com_displacement *= (_max - _min) + _min
        solver.set_links_COM_shift(com_displacement, obj_link_ids, env_idxs)

    def _randomize_mass(self, mass_range=(-0.1, 0.5), env_idxs=None):
        if self.object is None:
            return
        solver = self.rigid_solver
        env_idxs = env_idxs if env_idxs is not None else range(self.num_envs)
        _min, _max = mass_range
        obj_link_ids = self.object.coll_idxs_global
        n_links = len(obj_link_ids)
        mass_shift = gs.rand((len(env_idxs), n_links), dtype=float)
        mass_shift *= (_max - _min) + _min
        solver.set_links_mass_shift(mass_shift, obj_link_ids, env_idxs)

    def _random_force_torque(self, episode_length_buf):
        """ apply random external force on the object """
        if not self.rand_cfg.get('external_force', False) or self.object is None:
            return
        # only apply at probality of force_prob
        force_prob = self.rand_cfg.get('force_prob', 0.5)
        torand = torch.rand(self.num_envs, dtype=float) < force_prob

        pos = self.object.part_pos # (N, 2, 3)
        obj_link_idxs = self.object.coll_idxs_global
        # random sample around the object pos
        rand_force = torch.randn_like(pos)
        rand_force += pos
        force_scale = self.rand_cfg.get('force_scale', 5.0)
        # randomly scale it by +- 1.5:
        force_scales = torch.rand_like(rand_force) * 1.5 * force_scale
        rand_force *= force_scales.to(rand_force.device)
        rand_force[~torand, :] = 0.0
        self.rigid_solver.apply_links_external_force(
            force=rand_force, links_idx=obj_link_idxs,
        )

        # random torque
        rand_torque = torch.randn((self.num_envs, len(obj_link_idxs), 3), dtype=float)
        torque_scale = self.rand_cfg.get('torque_scale', 5.0)
        torque_scales = torch.rand_like(rand_torque) * 1.5 * torque_scale
        rand_torque *= torque_scales.to(rand_torque.device)
        rand_torque[~torand, :] = 0.0
        self.rigid_solver.apply_links_external_torque(
            torque=rand_torque, links_idx=obj_link_idxs
        )
