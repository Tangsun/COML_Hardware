# # import numpy as np

# if __name__ == "__main__":

#     import numpy as np
#     from scipy.spatial.transform import Rotation as R

#     # Generate a random rotation matrix
#     random_rotation = R.random().as_matrix()
#     print(random_rotation)

"""
TODO description.

Author: Spencer M. Richards
        Autonomous Systems Lab (ASL), Stanford
        (GitHub: spenrich)
"""

from tqdm.auto import tqdm
import pickle
from functools import partial
import time
import warnings
from math import pi, inf
import os
import argparse
import json
import matplotlib.pyplot as plt

from jax import config
# config.update("jax_debug_nans", True)

# # Parse command line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('seed', help='seed for pseudo-random number generation',
#                     type=int)
# parser.add_argument('M', help='number of trajectories to sub-sample',
#                     type=int)
# parser.add_argument('--use_x64', help='use 64-bit precision',
#                     action='store_true')
# parser.add_argument('--pnorm_init', help='set initial value for p-norm choices', type=float)
# parser.add_argument('--p_freq', help='set frequency for p-norm parameter update', type=float)
# parser.add_argument('--meta_epochs', help='set number of epochs for meta-training', type=int)
# parser.add_argument('--reg_P', help='set regularization for P matrix', type=float)
# parser.add_argument('--output_dir', help='set output directory', type=str)
# parser.add_argument('--hdim', help='number of hidden units per layer', type=int, default=32)
# args = parser.parse_args()

# # Set precision
# if args.use_x64:
#     os.environ['JAX_ENABLE_X64'] = 'True'

import jax                                          # noqa: E402
import jax.numpy as jnp                             # noqa: E402
from jax.example_libraries import optimizers             # noqa: E402
from dynamics import prior                          # noqa: E402
from utils import (tree_normsq, rk38_step, epoch,   # noqa: E402
                   odeint_fixed_step, random_ragged_spline, spline,
            params_to_cholesky, params_to_posdef, 
            quaternion_to_rotation_matrix, hat, vee)

# import jax.debug as jdebug

# def convert_p_qbar(p):
#     return jnp.sqrt(1/(1 - 1/p) - 1.1)

# def convert_qbar_p(qbar):
#     return 1/(1 - 1/(1.1 + qbar**2))

# # Initialize PRNG key
# key = jax.random.PRNGKey(args.seed)

# # Hyperparameters
# hparams = {
#     'seed':        args.seed,     #
#     'use_x64':     args.use_x64,  #
#     'num_subtraj': args.M,        # number of trajectories to sub-sample

#     # For training the model ensemble
#     'ensemble': {
#         'num_hlayers':    2,     # number of hidden layers in each model
#         'hdim':           args.hdim,    # number of hidden units per layer
#         'train_frac':     0.75,  # fraction of each trajectory for training
#         'batch_frac':     0.25,  # fraction of training data per batch
#         'regularizer_l2': 1e-4,  # coefficient for L2-regularization
#         'learning_rate':  1e-2,  # step size for gradient optimization
#         'num_epochs':     1000,  # number of epochs
#     },
#     # For meta-training
#     'meta': {
#         'num_hlayers':       2,          # number of hidden layers
#         'hdim':              args.hdim,         # number of hidden units per layer
#         'train_frac':        0.75,       #
#         'learning_rate':     1e-2,       # step size for gradient optimization
#         'num_steps':         args.meta_epochs,        # maximum number of gradient steps
#         'regularizer_l2':    1e-4,       # coefficient for L2-regularization
#         'regularizer_ctrl':  1e-3,       #
#         'regularizer_error': 0.,         #
#         'T':                 5.,         # time horizon for each reference
#         # 'T':                 20.,         # time horizon for each reference
#         'dt':                1e-2,       # time step for numerical integration
#         'num_refs':          10,         # reference trajectories to generate
#         'num_knots':         6,          # knot points per reference spline
#         'poly_orders':       (9, 9, 9),  # spline orders for each DOF
#         'deriv_orders':      (4, 4, 4),  # smoothness objective for each DOF
#         'min_step':          (-2., -2., -0.25),    #
#         'max_step':          (2., 2., 0.25),       #
#         'min_ref':           (-4.25, -3.5, 0.0),  #
#         'max_ref':           (4.5, 4.25, 2.0),     #
#         'p_freq':            args.p_freq,          # frequency for p-norm update
#         'regularizer_P':     args.reg_P,           # coefficient for P regularization
#     },
# }

if __name__ == "__main__":
    # os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="false"
    # os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"]=".50"
    # os.environ["XLA_PYTHON_CLIENT_ALLOCATOR"]="platform"

    # DATA PROCESSING ########################################################
    # Load raw data and arrange in samples of the form
    # `(t, x, u, t_next, x_next)` for each trajectory, where `x := (q,dq)`
    with open('data/2024-04-12_00-46-09_traj50_seed0.pkl', 'rb') as file:
        raw = pickle.load(file)
    num_dof = raw['q'].shape[-1]       # number of degrees of freedom
    param_dim = 2*num_dof + 9 + 3    # number of degrees of freedom including attitude (9 for rotation matrix, 3 for angular velocity)
    num_traj = raw['q'].shape[0]       # total number of raw trajectories
    num_samples = raw['t'].size - 1    # number of transitions per trajectory
    t = jnp.tile(raw['t'][:-1], (num_traj, 1))
    t_next = jnp.tile(raw['t'][1:], (num_traj, 1))
    x = jnp.concatenate((raw['q'][:, :-1], raw['dq'][:, :-1]), axis=-1)
    x_next = jnp.concatenate((raw['q'][:, 1:], raw['dq'][:, 1:]), axis=-1)
    u = raw['u'][:, :-1, :3]
    quat = raw['quat'][:, :-1]
    R = jax.vmap(jax.vmap(quaternion_to_rotation_matrix, in_axes=0), in_axes=0)(quat)
    R_flatten = R.reshape(R.shape[0], R.shape[1], -1)
    omega = raw['omega'][:, :-1]
    data = {'t': t, 'x': x, 'u': u, 'R_flatten': R_flatten, 'omega': omega, 't_next': t_next, 'x_next': x_next}

    # Shuffle and sub-sample trajectories
    # if hparams['num_subtraj'] > num_traj:
    #     warnings.warn('Cannot sub-sample {:d} trajectories! '
    #                   'Capping at {:d}.'.format(hparams['num_subtraj'],
    #                                             num_traj))
    #     hparams['num_subtraj'] = num_traj

    # key, subkey = jax.random.split(key, 2)
    # shuffled_idx = jax.random.permutation(subkey, num_traj)
    # hparams['subtraj_idx'] = shuffled_idx[:hparams['num_subtraj']]
    # data = jax.tree_util.tree_map(
    #     lambda a: jnp.take(a, hparams['subtraj_idx'], axis=0),
    #     data
    # )

    print(data['R_flatten'][25,1500].reshape((3,3)))