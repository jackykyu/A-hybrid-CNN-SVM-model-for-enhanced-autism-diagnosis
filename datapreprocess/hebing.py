import os
import numpy as np
import pandas as pd
import shutil

def read_npy_files_and_csv(npy_folder_path, csv_file_path, output_folder_path):
    # 创建输出文件夹
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 获取所有npy文件
    npy_files = [file for file in os.listdir(npy_folder_path) if file.endswith('.npy')]

    for npy_file in npy_files:
        # 读取npy文件
        npy_data = np.load(os.path.join(npy_folder_path, npy_file), allow_pickle=True).item()
        matrix = npy_data['matrix']
        output_1 = npy_data['output_1']
        output_2 = npy_data['output_2']
        group = npy_data['group']
        subid = npy_data['subid']

        # 读取csv文件
        csv_data = pd.read_csv(csv_file_path)
        csv_data['SUB_ID'] = csv_data['SUB_ID'].astype(str)
        sub_id_row = csv_data[csv_data['SUB_ID'] == subid]
        
        srs_values = sub_id_row[['SRS_RAW_TOTAL', 'SRS_AWARENESS', 'SRS_COGNITION', 'SRS_COMMUNICATION', 'SRS_MOTIVATION', 'SRS_MANNERISMS']].values.flatten()

        # 创建新的npy文件
        new_npy_data = {
            'matrix': matrix,
            'static': output_1,
            'dynamic': output_2,
            'srs': srs_values,
            'group': group,
            'subid': subid
        }
        new_npy_file = os.path.join(output_folder_path, f"{subid}.npy")
        np.save(new_npy_file, new_npy_data)

if __name__ == "__main__":
    npy_folder_path = "E:/ml/fmridata/"
    csv_file_path = "E:/ml/fmridata/normalized_output.csv"
    output_folder_path = "E:/ml/fmridata/datanew"

    read_npy_files_and_csv(npy_folder_path, csv_file_path, output_folder_path)
