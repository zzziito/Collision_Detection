import torch
import pandas as pd
import numpy as np
import os

def standardize_row_lengths(df, target_length=1000):
    standardized_data = []
 
    for _, row in df.iterrows():
        # NaN 값 제거 (선형 보간을 통해 채움)
        row = row.interpolate().fillna(method='bfill').fillna(method='ffill')
        current_length = len(row)
        if current_length < target_length:
            # 길이가 짧은 경우 interpolate
            x = np.linspace(0, current_length - 1, num=current_length)
            xp = np.linspace(0, current_length - 1, num=target_length)
            interpolated_row = np.interp(xp, x, row)
            standardized_data.append(interpolated_row)
        elif current_length > target_length:
            # 길이가 긴 경우 샘플링
            indices = np.linspace(0, current_length - 1, num=target_length, dtype=int)
            sampled_row = row.iloc[indices].values
            standardized_data.append(sampled_row)
        else:
            # 이미 길이가 1000인 경우
            standardized_data.append(row.values[:target_length])
 
    # 모든 행이 동일한 길이를 가지도록 DataFrame 생성
    standardized_df = pd.DataFrame(standardized_data, columns=range(target_length))
    return standardized_df

def extract_joint_from_filename(filename):
    parts = filename.split('_')
    if parts[1].startswith('joint'):
        joint_number = parts[2].split('.')[0]
        return joint_number
    return None

def load_and_combine_files(file_list, folder_path):
    combined_df = pd.DataFrame()
    for file in file_list:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        df = standardize_row_lengths(df)
        df.columns = ['Data' for _ in df.columns]
        
        joint_number = extract_joint_from_filename(file)
        df['joint_number'] = joint_number
        
        combined_df = pd.concat([combined_df, df], ignore_index=True)
    return combined_df

def prepare_dataset(df):
    df['label'] = df['label'].astype(int)
    df['joint_number'] = df['joint_number'].astype(int)
    
    # 클래스 레이블과 조인트 위치 추출
    labels = df['label'].values
    joints = df['joint_number'].values

    signals = df.drop(['label', 'joint_number'], axis=1).values
    signals = signals.astype(float)  

    return torch.tensor(signals, dtype=torch.float32), torch.tensor(labels, dtype=torch.int32), torch.tensor(joints, dtype=torch.int32)
