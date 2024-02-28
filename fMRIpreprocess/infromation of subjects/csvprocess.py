import pandas as pd
'''
#step 1 读取CSV文件
input_file1 = "F:/BIDE/statement/ABIDEI.csv"
input_file2 = "F:/BIDE/statement/ABIDEII.csv"
output_file1 = "F:/BIDE/statement/out1.csv"
output_file2 = "F:/BIDE/statement/out2.csv"

# 读取CSV文件并指定第一行为列名
df = pd.read_csv(input_file2, header=0)

# 筛选数据：SRS_RAW_TOTAL和SRS_AWARENESS均不为空的行 CSV1 CSV2栏的名字不同，全改成1的名字
df_filtered = df.dropna(subset=["SRS_RAW_TOTAL", "SRS_AWARENESS"])

# 筛选需要的列信息
selected_columns = ["SUB_ID", "SITE_ID", "DX_GROUP", "AGE_AT_SCAN", "SEX", "SRS_VERSION",
                    "SRS_RAW_TOTAL", "SRS_AWARENESS", "SRS_COGNITION", "SRS_COMMUNICATION",
                    "SRS_MOTIVATION", "SRS_MANNERISMS"]


df_selected = df_filtered[selected_columns]

# 保存筛选后的数据为新的CSV文件
df_selected.to_csv(output_file2, index=False)

print("筛选后的数据已保存为新的CSV文件。")
'''
'''
# step2 合并
file1 = "F:/BIDE/statement/out1.csv"
file2 = "F:/BIDE/statement/out2.csv"
df1 = pd.read_csv(file1)
df2 = pd.read_csv(file2)
# 合并两个DataFrame对象
merged_df = pd.concat([df1, df2], axis=0, ignore_index=True)

# 将"DX_GROUP"列中的元素1改为0，元素2改为1. 0 代表autism， 1代表control
merged_df["DX_GROUP"] = merged_df["DX_GROUP"].replace({1: 0, 2: 1})

# 保存合并后的DataFrame为新的CSV文件，假设合并后的文件名为merged_output.csv
merged_output_file = "F:/BIDE/statement/merged_output.csv"
merged_df.to_csv(merged_output_file, index=False)

print("两个CSV文件已合并并保存为新的CSV文件，并且\"DX_GROUP\"列的元素已修改。")'''

#step3 归一化某几列 并进行统计
import pandas as pd

# 读取CSV文件，假设文件名为input.csv，且位于当前工作目录下
input_file = "F:/BIDE/statement/merged_output.csv"

# 读取CSV文件并指定第一行为列名
df = pd.read_csv(input_file, header=0)

# 要进行归一化的列名列表
columns_to_normalize = ["SRS_RAW_TOTAL", "SRS_AWARENESS", "SRS_COGNITION", "SRS_COMMUNICATION",
                        "SRS_MOTIVATION", "SRS_MANNERISMS"]

# 对指定的列进行归一化（将数值映射到[0, 1]的范围）
df[columns_to_normalize] = (df[columns_to_normalize] - df[columns_to_normalize].min()) / (df[columns_to_normalize].max() - df[columns_to_normalize].min())

# 根据"SITE_ID"进行统计，计算每个SITE_ID对应的元素个数
site_id_counts = df["SITE_ID"].value_counts()

# 输出统计结果
print(site_id_counts)

columns_to_keep = ["SUB_ID", "DX_GROUP", "SRS_RAW_TOTAL", "SRS_AWARENESS", "SRS_COGNITION", "SRS_COMMUNICATION",
                   "SRS_MOTIVATION", "SRS_MANNERISMS"]
df_selected = df[columns_to_keep]
# 保存归一化后的数据为新的CSV文件
normalized_output_file = "F:/BIDE/statement/normalized_output.csv"
df_selected.to_csv(normalized_output_file, index=False)
print("新的归一化CSV文件已保存。")