from snapstack_msgs.msg import Wind
import jax
import jax.numpy as jnp

class WindSim():
    def __init__(self, key, num_traj, T, dt, wind_type):
        self.key = key
        self.num_traj = num_traj
        self.T = T
        self.dt = dt
        self.wind_type = wind_type
        self.t = jnp.arange(0, self.T + self.dt, self.dt)

        if self.wind_type is None or self.wind_type == 'None':
            self.winds = jnp.zeros((self.num_traj, len(self.t), 3))
        elif self.wind_type == 'sine':
            amplitude = 3
            vertical_shift = 10
            sine_wave = vertical_shift + amplitude * jnp.sin(self.t[:, None])

            direction_vector = jnp.array([1, 1, 1]) / jnp.sqrt(3)

            self.winds = jnp.array([sine_wave * direction_vector for i in range(self.num_traj)])

        elif self.wind_type == 'random_sine':
            keys = jax.random.split(self.key, 3)  # Split the key into parts for amplitude, period, and phase
        
            # Random amplitudes between 1 and 5
            amplitudes = 1.0 + jax.random.uniform(keys[0], shape=(self.num_traj, 3)) * 4
            
            # Random periods between 2*pi and 10.0
            periods = 2*jnp.pi + jax.random.uniform(keys[1], shape=(self.num_traj, 3)) * 8.0
            
            # Random phase shifts between 0 and 2*pi
            phases = jax.random.uniform(keys[2], shape=(self.num_traj, 3)) * 2 * jnp.pi
            
            # Calculate the sine wave for each trajectory and each dimension
            self.winds = jnp.array([amplitudes[i, :] * jnp.sin(2 * jnp.pi / periods[i, :] * self.t[:, None] + phases[i, :])
                            for i in range(self.num_traj)])
        elif type(self.wind_type) is int or float:
            self.winds = self.wind_type*jnp.ones((self.num_traj, len(self.t), 3))
        else:
            # Sample wind velocities from the training distribution
            self.w_min = 0.  # minimum wind velocity
            self.w_max = 6.  # maximum wind velocity
            self.a = 5.      # shape parameter `a` for beta distribution
            self.b = 9.      # shape parameter `b` for beta distribution
            self.key, subkey = jax.random.split(self.key, 2)
            self.w = self.w_min + (self.w_max - self.w_min)*jax.random.beta(subkey, self.a, self.b, (self.num_traj,))

            # Randomize wind direction
            random_vectors = jax.random.normal(self.key, (self.num_traj, 3))
            unit_vectors = random_vectors/jnp.linalg.norm(random_vectors, axis=1, keepdims=True)
            self.w_nominal_vectors = self.w[:, jnp.newaxis]*unit_vectors

            self.winds = jnp.repeat(self.w_nominal_vectors[:, jnp.newaxis, :], len(self.t), axis=1)
    
    def create_wind(self, w_nominal_vector):
        wind = Wind()

        wind.w_nominal.x = w_nominal_vector[0]
        wind.w_nominal.y = w_nominal_vector[1]
        # wind.w_nominal.y = 0
        wind.w_nominal.z = w_nominal_vector[2]
        # wind.w_nominal.z = 0

        wind.w_gust.x = 0
        wind.w_gust.y = 0
        wind.w_gust.z = 0

        return wind

    def generate_all_winds(self):
        all_winds = []
        for i in range(self.num_traj):
            wind_i = []
            for wind_t in self.winds[i]:
                wind_i.append(self.create_wind(wind_t))
            all_winds.append(wind_i)
        
        return all_winds