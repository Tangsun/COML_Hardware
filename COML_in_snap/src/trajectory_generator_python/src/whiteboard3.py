if __name__ == "__main__":
    import pickle
    import numpy as np

    with open('../data/training_data_2_1.pkl', 'rb') as file:
        raw = pickle.load(file)
    
    w_train = np.asarray(raw['w'])
    w_min, w_max = raw['w_min'], raw['w_max']
    a, b = raw['beta_params']


    for key, value in raw.items():
        print(f"{key}: {value}")

    print(raw['u'])

    error = raw['r'] - raw['q']
    derror = raw['dr'] - raw['dq']
    print(f"error: {error}")
    print(f"derror: {derror}")