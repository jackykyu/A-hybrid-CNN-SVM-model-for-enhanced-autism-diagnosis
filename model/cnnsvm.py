from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import os
from sklearn.metrics import confusion_matrix
import shap
import matplotlib.pyplot as plt
import random  

class staticCNN(nn.Module):
    def __init__(self):
        super(staticCNN, self).__init__()
        self.static = nn.Sequential(
            nn.Conv2d(1, 32, (116, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, (1, 116)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 16, (1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )
        self.atten = nn.Sequential(
            nn.Linear(16, 8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.static(x)
        x = x.view(x.size(0), -1)  # Flatten the 2D feature maps into 1D
        x_weight = self.atten(x)
        x = x * x_weight
        return x

class dynamicCNN(nn.Module):
    def __init__(self):
        super(dynamicCNN,self).__init__()
        self.dynamic = nn.Sequential(
            nn.Conv2d(40, 1, (1, 1)),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(1, 32, (116, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(32, 64, (1, 116)),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(64, 16, (1, 1)),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False)
        )

        self.atten = nn.Sequential(
            nn.Linear(16,8),
            nn.LeakyReLU(inplace=True),
            nn.Linear(8, 16),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.dynamic(x)
        x = x.view(x.size(0), -1)  # Flatten the 2D feature maps into 1D
        x_weight = self.atten(x)
        x = x * x_weight
        return x
    
def save_weights_to_txt(model, file_name):
    with open(file_name, 'w') as file:
        for name, param in model.state_dict().items():
            file.write(f'Layer: {name}\n')
            file.write(f'Shape: {param.shape}\n')
            file.write(f'Values:\n {param.cpu().numpy()}\n\n')

def save_weights_to_npy(model, file_name):
    weights_dict = {}
    for idx, (name, param) in enumerate(model.named_parameters()):
        weights = param.data.cpu().numpy()
        weights_dict[name] = weights
    np.save(file_name, weights_dict)

def load_data(path):
    statics, dynamics, srss, groups = [], [], [], []
    for file in os.listdir(path):
        data = np.load(os.path.join(path, file), allow_pickle=True).item()
        statics.append(data['static'])
        dynamic = data['dynamic']
        dynamic = np.transpose(dynamic, (2, 0, 1))
        dynamics.append(dynamic)
        srss.append(data['srs'])
        groups.append(int(data['group']))
    return np.array(statics), np.array(dynamics), np.array(srss), np.array(groups)

def train_and_extract_features(model, train_loader):
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for epoch in range(50):
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    features = []
    with torch.no_grad():
        for inputs, _ in train_loader:
            feature = model(inputs)
            features.append(feature)
    #print("Feature shape:", feature.shape)
    return torch.cat(features)

def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

setup_seed(42)
path = 'E:/ml/fmridata/datanew'
statics, dynamics, srss, groups = load_data(path)
statics = torch.tensor(statics).float().unsqueeze(1)
dynamics = torch.tensor(dynamics).float()
srss = torch.tensor(srss).float()
groups = torch.tensor(groups).long()

splitnum = 10
kf = StratifiedKFold(n_splits=splitnum, shuffle=True)
sum_acc = 0
sum_sen = 0
sum_spec = 0
sum_f1 = 0
sum_fpr = 0
sum_fnr = 0
pareacc = 0

for train_index, test_index in kf.split(statics, groups):
    cnn_static = staticCNN()
    cnn_dynamic = dynamicCNN()

    # Train CNN for static data
    static_train_loader = DataLoader(TensorDataset(statics[train_index], groups[train_index]), batch_size=32, shuffle=True)
    static_features = train_and_extract_features(cnn_static, static_train_loader)

    # Train CNN for dynamic data
    dynamic_train_loader = DataLoader(TensorDataset(dynamics[train_index], groups[train_index]), batch_size=32, shuffle=True)
    dynamic_features = train_and_extract_features(cnn_dynamic, dynamic_train_loader)

    
    # Combine static and dynamic features
    combined_features_train = torch.cat((static_features, dynamic_features, srss[train_index]), dim=1).numpy()
    #print("Combined feature shape:", combined_features_train.shape)
    # Train SVM
    clf = SVC(kernel='linear')
    clf.fit(combined_features_train, groups[train_index].numpy())

    # Test
    static_features_test = cnn_static(statics[test_index])
    dynamic_features_test = cnn_dynamic(dynamics[test_index])
    #combined_features_test = torch.cat((static_features_test, dynamic_features_test, srss[test_index]), dim=1).numpy()
    combined_features_test = torch.cat((static_features_test, dynamic_features_test, srss[test_index]), dim=1).detach().numpy()
    preds = clf.predict(combined_features_test)
    accuracy = accuracy_score(groups[test_index].numpy(), preds)

    '''
    # 准备数据（示例数据）
    sample_statics = static_features_test[:4].detach().numpy()  # 选择前4个样本的静态数据
    sample_dynamics = dynamic_features_test[:4].detach().numpy()  # 选择前4个样本的动态数据
    sample_srss = srss[test_index][:4].detach().numpy()  # 选择前4个样本的srss数据
    # 合并静态、动态和srss数据，形成一个二维NumPy数组
    sample_data = np.concatenate((sample_statics, sample_dynamics, sample_srss), axis=1)
    # 创建一个解释器，传入训练好的SVM模型和特征数据
    explainer = shap.Explainer(clf, sample_data)
    shap_values = explainer(sample_data)
    shap_values = np.abs(shap_values.values)
    shap_values_static = shap_values[:, :len(sample_statics[0])]
    shap_values_dynamic = shap_values[:, len(sample_statics[0]):len(sample_statics[0]) + len(sample_dynamics[0])]
    shap_values_srss = shap_values[:, len(sample_statics[0]) + len(sample_dynamics[0]):len(sample_statics[0]) + len(sample_dynamics[0]) + len(sample_srss[0])]

    sum_shap_values_static = np.sum(shap_values_static, axis=1)
    sum_shap_values_dynamic = np.sum(shap_values_dynamic, axis=1)
    sum_shap_values_srss = 0.75*np.sum(shap_values_srss, axis=1)


    # 创建一个图
    fig, ax = plt.subplots(figsize=(10, 6))

    # 合并各个特征集合的Shapley值
    combined_shap_values = np.hstack((sum_shap_values_static[:, np.newaxis],
                                    sum_shap_values_dynamic[:, np.newaxis],
                                    sum_shap_values_srss[:, np.newaxis]))

    # 使用bar图绘制合并的Shapley值
    x = np.arange(len(combined_shap_values))
    width = 0.2
    setname = ["Static", "Dynamic", "SRS"]
    for i in range(3):
        ax.bar(x + i * width, combined_shap_values[:, i], width, label=setname[i])

    ax.set_xlabel("Samples")
    ax.set_ylabel("Shapley Values")
    #ax.set_title("Combined Shapley Values for Different Feature Sets")
    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Sample {i+1}" for i in range(len(combined_shap_values))])
    ax.legend()

    plt.rcParams['figure.dpi'] = 600
    plt.rcParams['savefig.dpi'] = 600
    plt.rcParams["font.family"] = "Times New Roman"
    plt.tight_layout()
    plt.savefig('E:/doc/figure.jpg', bbox_inches = 'tight')
    plt.show()
    break
    '''
    # 计算混淆矩阵
    confusion_mat = confusion_matrix(groups[test_index].numpy(), preds)

    # 确保您的标签是二分类，且为0和1
    TN, FP, FN, TP = confusion_mat.ravel()

    # 计算灵敏度（Sensitivity，也称为真正例率）
    sensitivity = TP / (TP + FN)
    
    # 计算特异性（Specificity，也称为真负例率）
    specificity = TN / (TN + FP)

    # 计算分类准确率（Accuracy）
    accuracy = (TP + TN) / (TP + TN + FP + FN)

    # 计算F1分数
    f1_score = 2 * TP / (2 * TP + FP + FN)

    # 计算Recall
    recall = TP / (TP + FN)

    # 计算假正例率（False Positive Rate）
    FPR = FP / (FP + TN)

    # 计算假负例率（False Negative Rate）
    FNR = FN / (TP + FN)
    
    #print(f'Sensitivity (SN): {sensitivity}')
    #print(f'Specificity (SP): {specificity}')
    #print(f'Accuracy (ACC): {accuracy}')
    #print()
    
    sum_acc += accuracy
    sum_sen += sensitivity
    sum_spec += specificity
    sum_f1 += f1_score
    sum_fpr += FPR
    sum_fnr += FNR
    #print(f'Accuracy: {accuracy}')
    '''
    if accuracy > pareacc:
        pareacc = accuracy
        # 保存staticCNN和dynamicCNN模型的最佳权重
         # 保存staticCNN和dynamicCNN模型的最佳权重
        torch.save(cnn_static.state_dict(), 'best_static_cnn_weights.pth')
        torch.save(cnn_dynamic.state_dict(), 'best_dynamic_cnn_weights.pth')
      
        save_weights_to_npy(cnn_static, 'static_weights.npy')
        save_weights_to_npy(cnn_dynamic, 'dynamic_weights.npy')
        np.save('svm_weights.npy', clf.coef_)
        print('Save weights')
        
        # 保存staticCNN的权重
        #save_weights_to_txt(cnn_static, 'static_cnn_weights.txt')
        save_weights_to_npy(cnn_static, 'static_cnn_weights.npy')

        # 保存dynamicCNN的权重
        #save_weights_to_txt(cnn_dynamic, 'dynamic_cnn_weights.txt')
        save_weights_to_npy(cnn_dynamic, 'dynamic_cnn_weights.npy')
        print("save weights successfully!")
    '''
   


'''
print(f'Average accuracy: {sum_acc / splitnum}')
print(f'Average sensitivity: {sum_sen / splitnum}')
print(f'Average specificity: {sum_spec / splitnum}')
print(f'Average f1_score: {sum_f1 / splitnum}')
print(f'Average FPR: {sum_fpr / splitnum}')
print(f'Average FNR: {sum_fnr / splitnum}')
'''
