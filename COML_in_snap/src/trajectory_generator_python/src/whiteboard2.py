if __name__ == "__main__":
    import pickle
    from functools import partial
    import jax
    import jax.numpy as jnp
    from jax.experimental.ode import odeint
    from utils import spline, random_ragged_spline

    # Seed random numbers
    seed = 0
    key = jax.random.PRNGKey(seed)

    xmin_ = -4.25
    xmax_ =  4.5
    ymin_ = -3.5
    ymax_ =  4.25
    zmin_ =  0.0
    zmax_ =  2.0

    # Generate smooth trajectories
    T = 30
    num_traj = 50
    num_knots = 6
    # poly_orders = (9, 9, 9, 6)
    poly_orders = (9, 9, 9)
    # deriv_orders = (4, 4, 4, 2)
    deriv_orders = (4, 4, 4)
    # min_step = jnp.array([-2., -2., 0, -jnp.pi/6])
    min_step = jnp.array([-2., -2., -0.75])
    # max_step = jnp.array([2., 2., 2., jnp.pi/6])
    max_step = jnp.array([2., 2., 0.75])
    # min_knot = jnp.array([xmin_, ymin_, zmin_, -jnp.pi/3])
    min_knot = jnp.array([xmin_, ymin_, zmin_])
    # max_knot = jnp.array([xmax_, ymax_, zmax_, jnp.pi/3])
    max_knot = jnp.array([xmax_, ymax_, zmax_])

    key, *subkeys = jax.random.split(key, 1 + num_traj)
    subkeys = jnp.vstack(subkeys)
    in_axes = (0, None, None, None, None, None, None, None, None)
    t_knots, knots, coefs = jax.vmap(random_ragged_spline, in_axes)(
        subkeys, T, num_knots, poly_orders, deriv_orders,
        min_step, max_step, min_knot, max_knot
    )
    # x_coefs, y_coefs, ϕ_coefs = coefs
    r_knots = jnp.dstack(knots)

    # Sampled-time simulator
    @partial(jax.vmap, in_axes=(None, 0, 0))
    def simulate(ts, t_knots, coefs):
        """TODO: docstring."""
        # Construct spline reference trajectory
        def reference(t):
            # x_coefs, y_coefs, z_coefs, Ψ_coefs = coefs
            x_coefs, y_coefs, z_coefs = coefs
            x = spline(t, t_knots, x_coefs)
            y = spline(t, t_knots, y_coefs)
            z = spline(t, t_knots, z_coefs)
            # Ψ = spline(t, t_knots, Ψ_coefs)
            # Ψ = jnp.clip(Ψ, -jnp.pi/3, jnp.pi/3)
            # r = jnp.array([x, y, z, Ψ])
            x = jnp.clip(x, xmin_, xmax_)
            y = jnp.clip(y, ymin_, ymax_)
            z = jnp.clip(z, zmin_, zmax_)
            r = jnp.array([x, y, z])
            return r

        # Required derivatives of the reference trajectory
        def ref_derivatives(t):
            ref_vel = jax.jacfwd(reference)
            ref_acc = jax.jacfwd(ref_vel)
            r = reference(t)
            dr = ref_vel(t)
            ddr = ref_acc(t)
            return r, dr, ddr

        # Simulation loop
        def loop(carry, input_slice):
            t_prev = carry
            t = input_slice

            r, dr, ddr = ref_derivatives(t)
            carry = (t)
            output_slice = (r, dr, ddr)
            return carry, output_slice

        # Initial conditions
        t0 = ts[0]
        r0, dr0, ddr0 = ref_derivatives(t0)
        
        # Run simulation loop
        carry = (t0)
        carry, output = jax.lax.scan(loop, carry, ts[1:])
        r, dr, ddr = output

        # Prepend initial conditions
        r = jnp.vstack((r0, r))
        dr = jnp.vstack((dr0, dr))
        ddr = jnp.vstack((ddr0, ddr))

        return r, dr, ddr

    # Sample wind velocities from the training distribution
    w_min = 0.  # minimum wind velocity in inertial `x`-direction
    w_max = 6.  # maximum wind velocity in inertial `x`-direction
    a = 5.      # shape parameter `a` for beta distribution
    b = 9.      # shape parameter `b` for beta distribution
    key, subkey = jax.random.split(key, 2)
    w = w_min + (w_max - w_min)*jax.random.beta(subkey, a, b, (num_traj,))

    # Simulate tracking for each `w`
    dt = 0.01
    t = jnp.arange(0, T + dt, dt)  # same times for each trajectory
    # print('t_knots outside: ', t_knots.shape)
    r, dr, ddr = simulate(t, t_knots, coefs)

    # Take first trajectory
    r = r[5]
    dr = dr[5]
    ddr = ddr[5]

    print('r: ', r.shape)
    print('dr: ', dr.shape)
    print('ddr: ', ddr.shape)
    with jnp.printoptions(threshold=jnp.inf):
        print(r)