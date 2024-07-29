import re
import os
import pickle

def extract_metrics(filename):
    trials = []
    files = []
    metrics = []

    with open(filename, 'r') as file:
        lines = file.readlines()
        
        current_trial = None
        current_file = None
        current_metrics = {}

        for line in lines:
            if line.startswith('Trial Name:'):
                current_trial = line.split('Trial Name: ')[1].strip()
            
            elif line.startswith('File Name:'):
                current_file = line.split('File Name: ')[1].strip()
            
            elif 'Best Validation Epoch:' in line:
                current_metrics['Best Validation Epoch'] = int(re.findall(r'\d+', line)[0])
            elif 'Test Tracking Error:' in line:
                current_metrics['Test Tracking Error'] = float(re.findall(r'\d+\.?\d*', line)[0])
            elif 'Test Estimation Error:' in line:
                current_metrics['Test Estimation Error'] = float(re.findall(r'\d+\.?\d*', line)[0])
            
            elif '==========================================================' in line:
                if current_trial and current_file and current_metrics:
                    trials.append(current_trial)
                    files.append(current_file)
                    metrics.append(current_metrics)
                current_trial = None
                current_file = None
                current_metrics = {}

    return trials, files, metrics

def update_pickle_file(trial_name, metrics):
    # Search for the pickle file within the directory specified by trial_name
    for root, dirs, files in os.walk('./train_results/'):
        if trial_name in root:
            for file in files:
                if file.endswith('.pkl'):
                    pkl_path = os.path.join(root, file)
                    # Load existing data from the pickle file
                    with open(pkl_path, 'rb') as pkl_file:
                        data = pickle.load(pkl_file)
                    # Append the new metrics to the data
                    if 'metrics' not in data:
                        data['metrics'] = []
                    data['metrics'].append(metrics)
                    # Save the updated data back to the pickle file
                    with open(pkl_path, 'wb') as pkl_file:
                        pickle.dump(data, pkl_file)
                    print(f"Updated pickle file at: {pkl_path}")
                    return
    print(f"No pickle file found in directory for trial: {trial_name}")



# Use the functions to extract data from the specified file and update the pickle files
filename = 'total_analysis.txt'
trials, files, metrics = extract_metrics(filename)

for i in range(len(trials)):
    trial_name = trials[i]
    metric = metrics[i]
    update_pickle_file(trial_name, metric)
    print(f"Trial Name: {trials[i]}")
    print(f"File Name: {files[i]}")
    print("Performance:")
    print(f"\tBest Validation Epoch: {metrics[i]['Best Validation Epoch']}")
    print(f"\tTest Tracking Error: {metrics[i]['Test Tracking Error']}")
    print(f"\tTest Estimation Error: {metrics[i]['Test Estimation Error']}")
    print("==========================================================")
