import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import numpy as np



def getstat(confusion_matrix):
    # 计算真正例（True Positives）
    TP = confusion_matrix[1][1]

    # 计算真负例（True Negatives）
    TN = confusion_matrix[0][0]

    # 计算假正例（False Positives）
    FP = confusion_matrix[0][1]

    # 计算假负例（False Negatives）
    FN = confusion_matrix[1][0]

    # 计算准确率（Accuracy）
    accuracy = (TP+TN)/(TP+TN+FP+FN)

    # 计算精确率（Precision）
    precision = TP/(TP+FP)

    # 计算召回率（Recall，也称为灵敏度SEN）
    recall = TP/(TP+FN)

    # 计算特异性（Specificity，SPE）
    specificity = TN / (TN + FP)

    # 计算F1分数（F1 Score）
    f1 = 2 * precision * recall / (precision + recall)

    # 计算假正例率（False Positive Rate）
    FPR = FP / (FP + TN)

    # 计算假负例率（False Negative Rate）
    FNR = FN / (TP + FN)

    return accuracy, recall, specificity,  FPR, FNR, f1
    

'''
# 读取三个CSV文件
df1 = pd.read_csv('E:/ml/fmridata/statement/predicted_results.csv')
df2 = pd.read_csv('E:/ml/fmridata/statement/predicted_results1.csv')
df3 = pd.read_csv('E:/ml/fmridata/statement/predicted_results2.csv')

# 将三个DataFrame按照SUB_ID进行合并
merged_df = df1.merge(df2, on='SUB_ID').merge(df3, on='SUB_ID')

# 重命名列
merged_df = merged_df.rename(columns={'Predicted_Label_x': 'pre1', 'Predicted_Label_y': 'pre2', 'Predicted_Label': 'pre3'})

# 保存合并后的DataFrame到CSV文件
merged_df.to_csv('E:/ml/fmridata/statement/merged_file.csv', index=False)

# 读取两个CSV文件
df1 = pd.read_csv('E:/ml/fmridata/statement/merged_file.csv')
df2 = pd.read_csv('E:/ml/fmridata/statement/merged_output.csv')

# 根据SUB_ID合并两个DataFrame，并选择需要的列
merged_df = df1.merge(df2[['SUB_ID', 'SITE_ID', 'DX_GROUP', 'SEX','AGE_AT_SCAN']], on='SUB_ID')

# 保存合并后的DataFrame到CSV文件
merged_df.to_csv('E:/ml/fmridata/statement/finmerged_file.csv', index=False)
'''
# 读取CSV文件
df = pd.read_csv('E:/ml/fmridata/statement/finmerged_file_site.csv')

# 获取真实值和预测值
y_true = df['DX_GROUP']
y_pred1 = df['pre1']
y_pred2 = df['pre2']
y_pred3 = df['pre3']

'''
# 计算混淆矩阵
confusion_matrix1 = confusion_matrix(y_true, y_pred1)
confusion_matrix2 = confusion_matrix(y_true, y_pred2)
confusion_matrix3 = confusion_matrix(y_true, y_pred3)
accuracy1 = accuracy_score(y_true, y_pred1)
accuracy2 = accuracy_score(y_true, y_pred2)
accuracy3 = accuracy_score(y_true, y_pred3)
result_pre1 = getstat(confusion_matrix1)
result_pre2 = getstat(confusion_matrix2)
result_pre3 = getstat(confusion_matrix3)
results = np.mean([result_pre1, result_pre2, result_pre3], axis=0)
resltstd = np.std([result_pre1, result_pre2, result_pre3], axis=0)
print(f"accuracy, recall, specificity,  FPR, FNR, f1 for all:")
print(results)
print(resltstd)
'''

'''
# 获取性别信息
sex = df['SEX'].unique()  # 获取不同的性别类别


# 遍历不同的性别类别并计算混淆矩阵
for s in sex:
    subset_df = df[df['SEX'] == s]  # 根据性别筛选数据子集
    y_true_subset = subset_df['DX_GROUP']
    y_pred1_subset = subset_df['pre1']
    y_pred2_subset = subset_df['pre2']
    y_pred3_subset = subset_df['pre3']

    confusion_matrix1 = confusion_matrix(y_true_subset, y_pred1_subset)
    confusion_matrix2 = confusion_matrix(y_true_subset, y_pred2_subset)
    confusion_matrix3 = confusion_matrix(y_true_subset, y_pred3_subset)
    
    result_pre1 = getstat(confusion_matrix1)
    result_pre2 = getstat(confusion_matrix2)
    result_pre3 = getstat(confusion_matrix3)
    results = np.mean([result_pre1, result_pre2, result_pre3], axis=0)
    print(f"accuracy, recall, specificity,  FPR, FNR, f1 for SEX={s}:")
    print(results)
'''


# 获取SITE_ID信息
site_ids = df['SITE_ID'].unique()  # 获取不同的SITE_ID

# 遍历不同的SITE_ID并计算混淆矩阵
for site_id in site_ids:
    subset_df = df[df['SITE_ID'] == site_id]  # 根据SITE_ID筛选数据子集
    y_true_subset = subset_df['DX_GROUP']
    y_pred1_subset = subset_df['pre1']
    y_pred2_subset = subset_df['pre2']
    y_pred3_subset = subset_df['pre3']

    confusion_matrix1 = confusion_matrix(y_true_subset, y_pred1_subset)
    confusion_matrix2 = confusion_matrix(y_true_subset, y_pred2_subset)
    confusion_matrix3 = confusion_matrix(y_true_subset, y_pred3_subset)

    result_pre1 = getstat(confusion_matrix1)
    result_pre2 = getstat(confusion_matrix2)
    result_pre3 = getstat(confusion_matrix3)
    results = np.mean([result_pre1, result_pre2, result_pre3], axis=0)
    print(f"accuracy, recall, specificity,  FPR, FNR, f1 for SITE_ID={site_id}")
    print(results)

'''
# 划分年龄段
age_bins = [0, 10, 20, 30, float('inf')]  # 定义年龄段的边界
age_labels = ['0-10', '10-20', '20-30', '30+']  # 定义年龄段的标签

df['AGE_BIN'] = pd.cut(df['AGE_AT_SCAN'], bins=age_bins, labels=age_labels)
print(df['AGE_BIN'].value_counts())

# 获取不同年龄段的数据子集并计算混淆矩阵
for age_label in age_labels:
    subset_df = df[df['AGE_BIN'] == age_label]  # 根据年龄段筛选数据子集
    y_true_subset = subset_df['DX_GROUP']
    y_pred1_subset = subset_df['pre1']
    y_pred2_subset = subset_df['pre2']
    y_pred3_subset = subset_df['pre3']

    confusion_matrix1 = confusion_matrix(y_true_subset, y_pred1_subset)
    confusion_matrix2 = confusion_matrix(y_true_subset, y_pred2_subset)
    confusion_matrix3 = confusion_matrix(y_true_subset, y_pred3_subset)

    result_pre1 = getstat(confusion_matrix1)
    result_pre2 = getstat(confusion_matrix2)
    result_pre3 = getstat(confusion_matrix3)
    results = np.mean([result_pre1, result_pre2, result_pre3], axis=0)
    print(f"accuracy, recall, specificity,  FPR, FNR, f1 for AGE_BIN={age_label}")
    print(results)
'''
