import jax                                          # noqa: E402
import jax.numpy as jnp
from utils import hat, vee

import numpy as np
from scipy.spatial.transform import Rotation

# Generate a random rotation matrix


if __name__ == "__main__":
    # u_d = np.random.randn(3)
    u_d = np.array([0.15657147, 0.68898225, 0.79227353])
    # random_rotation = Rotation.random()
    # R_matrix = random_rotation.as_matrix()
    R_matrix = np.array([[ 0.1362983, -0.75171602,  0.64524862],
                [ 0.86056514,  0.41250012,  0.298783],
                [-0.4907651,   0.51455483,  0.7031237 ]])
    R_flatten = R_matrix.flatten()
    # Omega = np.random.rand(3)
    Omega = np.array([0.59286713, 0.88477115, 0.5278033 ])

    R = R_flatten.reshape((3,3))

    f_d = jnp.linalg.norm(u_d)
    u_d = jnp.array([0, 0, 1])
    b_3d = -u_d / jnp.linalg.norm(u_d)
    b_1d = jnp.array([1, 0, 0])
    cross = jnp.cross(b_3d, b_1d)
    b_2d = cross / jnp.linalg.norm(cross)

    R_d = jnp.column_stack((jnp.cross(b_2d, b_3d), b_2d, b_3d))

    print(R_d)

    Omega_d = jnp.array([0, 0, 0])
    dOmega_d = jnp.array([0, 0, 0])

    k_R = jnp.array([1400.0, 1400.0, 1260.0])/1000.0
    k_Omega = jnp.array([330.0, 330.0, 300.0])/1000.0
    J = jnp.diag(jnp.array([0.03, 0.03, 0.09]))

    e_R = 0.5 * vee(R_d.T@R - R.T@R_d)
    e_Omega = Omega - R.T@R_d@Omega_d

    print('e_R:', e_R)
    print('e_Omega:', e_Omega)

    M = - k_R*e_R \
        - k_Omega*e_Omega \
        + jnp.cross(Omega, J@Omega) \
        - J@(hat(Omega)@R.T@R_d@Omega_d - R.T@R_d@dOmega_d)

    dOmega = jax.scipy.linalg.solve(J, M - jnp.cross(Omega, J@Omega), assume_a='pos')
    dR = R@hat(Omega)
    dR_flatten = dR.flatten()

    e_3 = jnp.array([0, 0, 1])
    u = -f_d*R@e_3

    print(dOmega)
    print(dR)