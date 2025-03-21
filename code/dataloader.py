import os
import numpy as np
import re
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from itertools import combinations


'''
# 计算边和权重
def compute_and_merge_edges_weights(left_coords, right_coords, left_start_idx=0, right_start_idx=8, scale_factor=100):
    def compute_edges_and_weights(coords, start_idx=0):
        edges = []
        weights = []
        for i, j in combinations(range(len(coords)), 2):  # All pair combinations
            dist = np.linalg.norm(coords[i] - coords[j])  # Euclidean distance
            weight = 1 / dist if dist > 0 else 0  # Inverse distance as weight
            edges.append([start_idx + i, start_idx + j])
            weights.append(weight)
        edges = torch.tensor(edges, dtype=torch.long).t()  # Transpose for PyG format
        weights = torch.tensor(weights, dtype=torch.float)
        return edges, weights

        # 分别计算左右脚的边和权重

    left_edges, left_weights = compute_edges_and_weights(left_coords, start_idx=left_start_idx)
    right_edges, right_weights = compute_edges_and_weights(right_coords, start_idx=right_start_idx)

    # 合并左右脚的边和权重
    edges = torch.cat([left_edges, right_edges], dim=1)
    weights = torch.cat([left_weights, right_weights])

    # 放大权重
    weights = weights * scale_factor

    return edges, weights
'''

def process_dataset(data_path, label_file_path):
    """
    处理数据集路径下符合特定命名规则的所有TXT文件，将其处理成形状为 n×120×16×100 的数据，并从标签文件中获取相应标签。

    参数：
    - data_path: 数据集路径，包含多个TXT文件。
    - label_file_path: 标签文件路径，包含每个文件的标签信息。

    返回：
    - 数据集的numpy数组，形状为 n×120×16×100，以及相应的标签，和mask。
    """
    all_data = []
    all_labels = []
    all_masks = []
    all_files = []
    i = 0

    # 读取标签文件
    labels_df = pd.read_csv(label_file_path)

    # 获取路径下所有符合目标命名规则的txt文件
    file_list = [f for f in os.listdir(data_path)
                 if re.match(r'^(Ga|Ju|Si)(Co|Pt)\d{2}_\d{2}\.txt$', f)]

    for file_name in file_list:
        # 构造文件路径
        file_path = os.path.join(data_path, file_name)

        # 读取文件数据
        data = np.loadtxt(file_path)

        # 计算需要提取的有效数据点数
        num_data_points = len(data)

        # 如果数据点不够12000，取最后可以被100整除的数据点
        if num_data_points < 12000:
            data_to_take = (num_data_points // 100) * 100  # 取能够被100整除的最大值
            signals = data[-data_to_take:, 1:17]  # 取最后有效数据点，选择第2到第17列的16个通道数据
            signals = np.pad(signals, ((0, 12000 - data_to_take), (0, 0)), mode='constant')  # 填充至12000个数据点
        else:
            signals = data[-12000:, 1:17]  # 取最后12000个数据点，选择第2到第17列的16个通道数据

        # 每100个数据点为一组，重塑为 (120, 100, 16)，然后转置为 (120, 16, 100)
        reshaped_signals = signals.reshape(120, 100, 16).transpose(0, 2, 1)

        # 生成mask，填充的部分标为False
        mask = np.zeros(120, dtype=bool)  # 默认mask为True，表示需要遮盖
        if num_data_points < 12000:
            mask[-(120-data_to_take//100):] = True  # 将填充部分标记为False
        if np.all(mask):
            print(file_name)

        # 根据文件名获取标签
        subject_id = file_name.split('_')[0]  # 获取文件名（去除扩展名）

        # 从标签文件中查找对应的标签
        label_row = labels_df[labels_df['ID'] == subject_id]
        if not label_row.empty:
            group = label_row['Group'].values[0]
            # 将标签为2的变为0
            group = 0 if group == "CO" else 1

            # 将数据、标签和mask一起添加到列表
            all_data.append(reshaped_signals)
            all_labels.append(group)
            all_masks.append(mask)
            all_files.append(file_name)

    # 将所有文件的数据合并成一个大的numpy数组，形状为 n×120×16×100
    all_data = np.stack(all_data, axis=0)
    all_labels = np.array(all_labels)
    all_masks = np.stack(all_masks, axis=0)
    all_files = np.array(all_files, dtype=object)

    return all_data, all_labels, all_masks, all_files


def get_foot_graph_edges_and_weights(left_coords, right_coords, left_start_idx=0, right_start_idx=8, scale_factor=100):  
    """  
    计算左右脚传感器的边与权重，并合并为无向图。  
    
    参数:  
        left_coords (ndarray): 左脚传感器坐标数组，形状为 (8, 2)。  
        right_coords (ndarray): 右脚传感器坐标数组，形状为 (8, 2)。  
        left_start_idx (int): 左脚节点的全局起始索引（默认为0）。  
        right_start_idx (int): 右脚节点的全局起始索引（默认为8）。  

    返回:  
        edges (Tensor): 合并后的边张量，形状为 (2, num_edges)。  
        weights (Tensor): 边对应的权重张量，形状为 (num_edges,)。  
    """  

    # 定义通用的边与权重计算函数  
    def compute_edges_and_weights(coords, start_idx=0):  
        edges = []  
        weights = []  

        # 区域划分  
        regions = {  
            1: [0, 1, 2],  # 后脚部 (1, 2, 3号传感器)  
            2: [3, 4, 5, 6],  # 中脚部 (4, 5, 6, 7号传感器)  
            3: [7]  # 前脚部 (8号传感器)  
        }  

        # 区域内全连接  
        for region in regions.values():  
            for i, j in combinations(region, 2):  # 全连接  
                dist = np.linalg.norm(coords[i] - coords[j])  # 计算距离  
                weight = 1 / dist if dist > 0 else 0  # 距离倒数作为权重  
                edges.append([start_idx + i, start_idx + j])  
                edges.append([start_idx + j, start_idx + i])  # 无向图需要双向边  
                weights.extend([weight, weight])  

        # 跨区域功能性连接  
        functional_connections = [  
            (2, 4),  # 区域1 → 区域2功能性连接：2和4相连  
            (3, 5),  # 区域1 → 区域2功能性连接：3和5相连  
            (7, 8),  # 区域2 → 区域3功能性连接：7和8相连  
            (6, 8)   # 区域2 → 区域3功能性连接：6和8相连  
        ]  
        for i, j in functional_connections:  
            i -= 1  # 从1-based转为0-based索引  
            j -= 1  
            dist = np.linalg.norm(coords[i] - coords[j])  # 计算距离  
            weight = 1 / dist if dist > 0 else 0  # 距离倒数作为权重  
            edges.append([start_idx + i, start_idx + j])  
            edges.append([start_idx + j, start_idx + i])  # 无向图需要双向边  
            weights.extend([weight, weight])  

        edges = torch.tensor(edges, dtype=torch.long).t()  # 转为 PyG 格式  
        weights = torch.tensor(weights, dtype=torch.float)
        weights = weights * scale_factor
        return edges, weights

    left_edges, left_weights = compute_edges_and_weights(left_coords, start_idx=left_start_idx)
    right_edges, right_weights = compute_edges_and_weights(right_coords, start_idx=right_start_idx)

    # 合并左右脚的边和权重（左右脚独立，不相连）
    edges = torch.cat([left_edges, right_edges], dim=1)
    weights = torch.cat([left_weights, right_weights], dim=0)

    return edges, weights


'''
def get_foot_graph_edges_and_weights_knn(left_coords, right_coords, k=3, left_start_idx=0, right_start_idx=8,
                                         scale_factor=100):
    """  
    使用KNN方法构建左右脚的边与权重，并合并为无向图。  

    参数:  
        left_coords (ndarray): 左脚传感器坐标数组，形状为 (8, 2)。  
        right_coords (ndarray): 右脚传感器坐标数组，形状为 (8, 2)。  
        k (int): 每个节点的邻居数量（默认3）。  
        left_start_idx (int): 左脚节点的全局起始索引（默认为0）。  
        right_start_idx (int): 右脚节点的全局起始索引（默认为8）。  

    返回:  
        edges (Tensor): 合并后的边张量，形状为 (2, num_edges)。  
        weights (Tensor): 边对应的权重张量，形状为 (num_edges,)。  
    """

    def compute_edges_and_weights_knn(coords, start_idx=0, k=3):
        """  
        使用KNN方法计算节点的边和权重。  

        参数:  
            coords (ndarray): 节点坐标，形状为 (num_nodes, 2)。  
            start_idx (int): 当前脚的节点索引起点，用于全局索引。  
            k (int): 每个节点的邻居数量。  

        返回:  
            edges (Tensor): 当前部分的边张量，形状为 (2, num_edges)。  
            weights (Tensor): 对应的权重张量，形状为 (num_edges,)。  
        """
        num_nodes = coords.shape[0]
        edges = []
        weights = []

        # 计算节点之间的距离矩阵  
        dist_matrix = distance.cdist(coords, coords, metric='euclidean')

        # 对每个节点找到k最近邻  
        for i in range(num_nodes):
            # 获取当前节点到其他节点的距离，并排序（排除自身）  
            neighbors = np.argsort(dist_matrix[i])[:k + 1]  # 获取前k+1个（包括自身）  
            neighbors = neighbors[neighbors != i]  # 排除自身  

            for neighbor in neighbors:
                # 构建双向边  
                dist = dist_matrix[i, neighbor]  # 获取距离  
                weight = 1 / dist if dist > 0 else 0  # 距离的倒数为权重  
                edges.append([start_idx + i, start_idx + neighbor])  # i -> neighbor  
                edges.append([start_idx + neighbor, start_idx + i])  # neighbor -> i  
                weights.extend([weight, weight])

                # 转为 PyG 格式  
        edges = torch.tensor(edges, dtype=torch.long).t()
        weights = torch.tensor(weights, dtype=torch.float)
        return edges, weights

        # 分别计算左右脚的边和权重  

    left_edges, left_weights = compute_edges_and_weights_knn(left_coords, start_idx=left_start_idx, k=k)
    right_edges, right_weights = compute_edges_and_weights_knn(right_coords, start_idx=right_start_idx, k=k)

    # 合并左右脚的边和权重（左右脚独立，不相连）  
    edges = torch.cat([left_edges, right_edges], dim=1)
    weights = torch.cat([left_weights, right_weights], dim=0)
    weights = weights * scale_factor

    return edges, weights


# 调用函数获取边和权重  
edges, weights = get_foot_graph_edges_and_weights_knn(left_coords, right_coords, k=3)

# 输出结果  
print("Edges:")
print(edges)  # 64, 96
print("\nWeights:")
print(weights)
'''


class Load_Dataset(Dataset):
    def __init__(self, data, labels, masks, device):
        """
        初始化数据集。

        参数：
        - data: numpy 数组，形状为 (n, 120, 16, 100)，表示 n 个样本。
        - labels: numpy 数组，形状为 (n,)，表示每个样本的标签。
        - masks: numpy 数组，形状为 (n, 120)，表示每个样本的 mask。
        """
        self.data = torch.tensor(data, dtype=torch.float32).to(device)  # 转换为 Tensor
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)  # 转换为 Tensor
        self.masks = torch.tensor(masks, dtype=torch.bool).to(device)  # 转换为 Tensor

    def __len__(self):
        """
        返回数据集的大小，即样本的数量。
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        根据索引返回单个样本及其对应的标签和 mask。

        参数：
        - idx: 数据的索引。

        返回：
        - 样本数据 (形状为 (120, 16, 100))
        - 标签 (标量)
        - mask (形状为 (120,))，bool 类型
        """
        sample = self.data[idx]
        label = self.labels[idx]
        mask = self.masks[idx]

        return sample, label, mask


def data_generator(data_path, label_file_path, batch_size, shuffle, device, save_path, random_seed=42):
    """
    数据生成器，使用随机划分将数据划分为训练集和测试集，并创建对应的 DataLoader。

    参数：
    - data_path: 数据集路径，包含多个TXT文件。
    - label_file_path: 标签文件路径，包含每个文件的标签信息。
    - batch_size: 批处理的大小。
    - shuffle: 是否打乱数据。
    - device: 数据加载设备（如 'cuda' 或 'cpu'）。
    - save_path: 保存训练集和测试集的路径。
    - test_size: 测试集划分比例（默认 0.2）。
    - random_seed: 随机种子，保证可复现（默认 42）。

    返回：
    - 训练集和测试集的 DataLoader。
    """
    # 获取数据集、标签和掩码
    data, labels, masks, files = process_dataset(data_path, label_file_path)

    # 打乱数据、标签和掩码（由 train_test_split 处理，没有 shuffle 参数时按需要手动打乱）
    train_data, temp_data, train_labels, temp_labels, train_masks, temp_masks, train_files, temp_files = train_test_split(
        data, labels, masks, files, test_size=0.2, random_state=random_seed, shuffle=shuffle
    )

    # 第二次划分：进一步将验证+测试集 (20%) 分为验证集 (10%) 和测试集 (10%)
    val_data, test_data, val_labels, test_labels, val_masks, test_masks, val_files, test_files = train_test_split(
        temp_data, temp_labels, temp_masks, temp_files, test_size=0.5, random_state=random_seed, shuffle=shuffle
    )

    # 保存训练集和测试集到 npy 文件
    np.save(f"{save_path}/train_data.npy", train_data)
    np.save(f"{save_path}/train_labels.npy", train_labels)
    np.save(f"{save_path}/train_masks.npy", train_masks)
    np.save(f"{save_path}/train_files.npy", train_files)

    np.save(f"{save_path}/test_data.npy", test_data)
    np.save(f"{save_path}/test_labels.npy", test_labels)
    np.save(f"{save_path}/test_masks.npy", test_masks)
    np.save(f"{save_path}/test_files.npy", test_files)

    # 创建 Dataset 对象
    train_dataset = Load_Dataset(train_data, train_labels, train_masks, device)
    test_dataset = Load_Dataset(test_data, test_labels, test_masks, device)

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader