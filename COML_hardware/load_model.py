import os
import pickle
import jax.numpy as jnp
import numpy as np
from dynamics import prior
from utils import params_to_posdef

def convert_p_qbar(p):
    return np.sqrt(1/(1 - 1/p) - 1.1)


if __name__ == "__main__":
    trial_name = 'hardware_13'
    filename = 'seed=0_M=50_E=1000_pinit=2.00_pfreq=2000_regP=0.0000.pkl'
    model_dir = f'train_results/{trial_name}'

    model_pkl_loc = os.path.join(model_dir, filename)
    with open(model_pkl_loc, 'rb') as f:
        train_results = pickle.load(f)
    
    test_params = {}
    test_params['pnorm'] = convert_p_qbar(train_results['pnorm'])
    test_params['W'] = train_results['model']['W']
    test_params['b'] = train_results['model']['b']
    test_params['Λ'] = params_to_posdef(train_results['controller']['Λ'])
    test_params['K'] = params_to_posdef(train_results['controller']['K'])
    test_params['P'] = params_to_posdef(train_results['controller']['P'])

    def adaptation_law(q, dq, R_flatten, Omega, r, dr, params=test_params):
        # Regressor features
        y = jnp.concatenate((q, dq, R_flatten, Omega))
        for W, b in zip(params['W'], params['b']):
            y = jnp.tanh(W@y + b)

        # Auxiliary signals
        Λ, P = params['Λ'], params['P']
        e, de = q - r, dq - dr
        s = de + Λ@e

        dA = jnp.outer(s, y) @ P
        return dA, y

    def controller(q, dq, r, dr, ddr, f_hat, params=test_params, type='adaptive'):
        if type == 'adaptive':
            # Auxiliary signals
            Λ, K = params['Λ'], params['K']

            e, de = q - r, dq - dr
            s = de + Λ@e
            v, dv = dr - Λ@e, ddr - Λ@de

            # Control input and adaptation law
            H, C, g, B = prior(q, dq)
            τ = H@dv + C@v + g - f_hat - K@s
            u_d = jnp.linalg.solve(B, τ)
            # jdb.print('{}', u_d)
            return u_d, τ
    
    def loop(t, q, dq, R_flatten, Omega, r, dr, ddr, params=test_params):
        pA_prev = ...
        t_prev = ...
        dA_prev = ...
        qn = 1.1 + params['pnorm']**2

        # Integrate adaptation law via trapezoidal rule
        dA, y = adaptation_law(q, dq, R_flatten, Omega, r, dr)
        pA = pA_prev + (t - t_prev)*(dA_prev + dA)/2
        P = params['P']
        A = (jnp.maximum(jnp.abs(pA), 1e-6 * jnp.ones_like(pA))**(qn-1) * jnp.sign(pA) * (jnp.ones_like(pA) - jnp.isclose(pA, 0, atol=1e-6)) ) @ P
            
        f_hat = A @ y
        u_d, τ = controller(q, dq, r, dr, ddr, f_hat, type='adaptive')