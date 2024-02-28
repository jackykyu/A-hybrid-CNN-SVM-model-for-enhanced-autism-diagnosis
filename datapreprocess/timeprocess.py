import os
import csv
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import morlet
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import gc

# 读取CSV文件中的subid和group列元素
csv_file = 'E:/ml/code/data/normalized_output.csv'
txt_folder = 'E:/ml/code/fmridata/'

subid_group = {}  # 存储subid和对应的group

with open(csv_file, 'r') as file:
    reader = csv.DictReader(file)
    for row in reader:
        subid = row['SUB_ID']
        group = row['DX_GROUP']
        subid_group[subid] = group


data_objects = []  # 存储处理后的数据对象

# 读取文件夹下的所有TXT文件
txt_files = [f for f in os.listdir(txt_folder) if f.endswith('.txt')]

# 处理每个TXT文件， 得到matrix和对应的label/group
for txt_file in txt_files:
    txt_file_path = os.path.join(txt_folder, txt_file)
    filename = os.path.splitext(txt_file)[0]  # 去除文件扩展名
    for subid, group in subid_group.items():
        if subid in filename:
            # 执行与group分配相关的操作，例如读取矩阵并分配group
            with open(txt_file_path, 'r') as file:
                lines = file.readlines()
                lines = lines[1:]  # 读取除第一行外的所有行
                matrix = []
                for line in lines:
                    row = line.strip().split()  # 按空格切分元素
                    matrix.append(row)
                matrix = np.array(matrix, dtype=float)

                # 创建数据对象，存储矩阵和对应的group
                data_object = {
                    'matrix': matrix,
                    'group': group,
                    'subid': subid
                }
                data_objects.append(data_object)
            break

""" 
first_object = data_objects[0]  # 获取第一个对象的形状
matrix_shape = len(first_object['matrix']), len(first_object['matrix'][0])
group_value = first_object['group']
print("Matrix Shape:", matrix_shape)
print("Group Value:", group_value) 
"""

# 计算static functional connectivity 皮尔逊相关系数
def process_matrix_Pearson(matrix):
    # 检查并处理矩阵中的NaN值
    matrix = np.nan_to_num(matrix, nan=0.0)
    
    # 计算皮尔逊相关系数
    corr_matrix = np.corrcoef(matrix, rowvar=False)
    
    return corr_matrix

# 对每列进行高斯权重求和
def weighted_sum(matrix):
    m, n = matrix.shape
    result = np.zeros((m, 1))
    sigma = int(np.ceil(int(n)**0.5 * 2))
    for i in range(m):
        row = matrix[i]
        weights = gaussian_filter1d(np.ones(n), sigma)
        weights /= np.sum(weights)  # Normalize weights to ensure sum equals 1
        result[i] = np.dot(row, weights)

    return result

# PCA降维，矩阵降至一维
def matrix_pca_reduction(matrix):
    # 创建PCA对象
    pca = PCA(n_components=1)

    # 对矩阵进行降维
    reduced_matrix = pca.fit_transform(matrix)

    return reduced_matrix.flatten()



# 计算小波相干谱，返回值为num_bins × len(a)
def calculate_wavelet_coherence(a, b, num_bins=40):
    # 小波参数设置
    sampling_rate = 1  # 假设采样率为1Hz
    time = np.arange(len(a)) / sampling_rate
    freq_range = (0.01, 0.08)
    freqs = np.linspace(freq_range[0], freq_range[1], 40)

    # 连续小波变换
    coeffs_a = continuous_wavelet_transform(a, time, freqs)
    coeffs_b = continuous_wavelet_transform(b, time, freqs)

    # 计算小波相干谱
    coherence_matrix = np.abs(coeffs_a * np.conj(coeffs_b)) ** 2

    return coherence_matrix

# 使用morlet小波
def continuous_wavelet_transform(signal, time, freqs):
    coeffs = np.zeros((len(freqs), len(signal)), dtype=complex)
    for i, f in enumerate(freqs):
        wavelet = morlet(len(signal), f)
        coeffs[i] = np.convolve(signal, wavelet, mode='same')

    return coeffs

'''
# 输入两个列向量 a 和 b
a = np.random.rand(100)/10
b = np.random.rand(100)/10

# 计算小波相干谱
coherence_matrix = calculate_wavelet_coherence(a, b)
print(coherence_matrix.shape)


# 绘制小波相干谱图
plt.imshow(coherence_matrix, aspect='auto', origin='lower', cmap='jet', extent=[0, len(a), 0.01, 0.08])


plt.colorbar(label='Magnitude')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.title('Wavelet Coherence')
plt.show()
'''

#
def process_dynamic_matrix(matrix, l):
    m, n = matrix.shape
    matrix = np.nan_to_num(matrix, nan=0.0)
    result = np.zeros((n, n,l))

    for i in range(n):
            for j in range(n):
                if i != j :
                    tempmat = calculate_wavelet_coherence(matrix[:, i], matrix[:, j],l)
                    tempvec = weighted_sum(tempmat)
                    for k in range(l):
                        result[i,j,k] = tempvec[k]

    return result
    
def process_dynamic_matrix_raw(matrix, l):
    num_cols= matrix.shape[1]
    output_matrix = np.empty((num_cols, num_cols), dtype=object)

    for i in range(num_cols):
            for j in range(i,num_cols):
                col1 = matrix[:, i]
                col2 = matrix[:, j]
                tempmat = calculate_wavelet_coherence(col1,col2,l)
                if i==j:
                    output_matrix[i, j] = tempmat
                else:
                    output_matrix[i, j] = tempmat
                    output_matrix[j, i] = tempmat

    return output_matrix


data_objects_with_results = []  # 存储包含结果的数据对象

'''
sampledata = data_objects[0]
matrix = sampledata['matrix']
group = sampledata['group']
subid = sampledata['subid']
print(matrix.shape) 
col1 = matrix[:, 0]
col2 = matrix[:, 0]
tempmat = calculate_wavelet_coherence(col1,col2,40)
print(tempmat.shape) # (40,n)对每两个区域，计算了40个频率的小波相干谱，每个频率下有n个时间点的值
'''
startindex=0

for data_object in data_objects:
        subid = int(data_object['subid'])
        matrix = data_object['matrix']
        group = data_object['group']
        subid = data_object['subid'] 
        group = str(int(group)-1)  # 0 ASD 1 TC
        file_path = 'E:/ml/code/data'
        file_name = f"{subid}.npy"
        full_path = f"{file_path}/{file_name}"
        
        try:
            # 调用第一个函数处理矩阵，并获取输出结果
            output_1 = process_matrix_Pearson(matrix)

            # 调用第二个函数处理矩阵，并获取输出结果
            output_2 = process_dynamic_matrix_raw(matrix, l = 40)

            # 创建包含结果的数据对象
            data_object_with_results = {
                'matrix': matrix,
                'output_1': output_1,
                'output_2': output_2,
                'group': group,
                'subid': subid
            }
            np.save(full_path, data_object_with_results)
            print("finish 1 !")
            #data_objects_with_results.append(data_object_with_results)
            del data_object_with_results
            gc.collect()
        except:
            print(subid)

totalpath = "E:/ml/code/data"
#np.save(totalpath,data_objects_with_results)
# 打印第一个对象的结果
#first_object_with_results = data_objects_with_results[0]
#matrix_shape = len(first_object_with_results['matrix']), len(first_object_with_results['matrix'][0])
#group_value = first_object_with_results['group']
#output_1_value = first_object_with_results['output_1']
#output_2_value = first_object_with_results['output_2']

#print("Matrix Shape:", matrix_shape)
#print("Group Value:", group_value)
#print("done")


