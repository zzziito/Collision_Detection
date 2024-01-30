# Generate target dataset using Forward Kinematics

import pandas as pd
import numpy as np
import os

from utils import compute

def calculate_tau_and_save(input_csv_path, output_csv_path):
    input_data = pd.read_csv(input_csv_path)
    tau_values = []

    for _, row in input_data.iterrows():
        q_values = row.values
        
        for i in range(1, len(q_values)):
            q_current = q_values[i]
            q_prev = q_values[i-1]
            
            x_current = compute.compute_xc(q_values)
            J = compute.compute_jacobian(q_values, 3, 7) 

        x_current = compute.compute_xc(q_values)
        J = compute.compute_jacobian(q_values, 3, 7) 
        F = np.random.rand(3) 

        tau = J.T.dot(F) - 100 * (q_current - q_prev)
        
        tau_values.append(tau)

    tau_df = pd.DataFrame(tau_values)
    tau_df.to_csv(output_csv_path, index=False)

def process_all_files(input_folder, target_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv') and 'fre_joint_' in f]

    for file in files:
        input_csv_path = os.path.join(input_folder, file)
        output_csv_path = os.path.join(target_folder, file)
        
        calculate_tau_and_save(input_csv_path, output_csv_path)
        
input_folder = "/home/rtlink/robros/dataset/robros_dataset/input"
target_folder = "/home/rtlink/robros/dataset/robros_dataset/target"

if __name__=="__main__":
    process_all_files(input_folder, target_folder)

    