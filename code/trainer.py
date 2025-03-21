import torch
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
from dataloader import get_foot_graph_edges_and_weights
import os
import pandas as pd
import numpy as np


# 定义传感器的坐标
left_coords = np.array([
    [-500, -800], [-700, -400], [-300, -400], [-700, 0],
    [-300, 0], [-700, 400], [-300, 400], [-500, 800]
])
right_coords = np.array([
    [500, -800], [700, -400], [300, -400], [700, 0],
    [300, 0], [700, 400], [300, 400], [500, 800]
])


class Trainer:
    def __init__(self, model, num_epochs, batch_size, lr, train_loader, test_loader, criterion, device,
                 save_path):
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

        # 数据加载器
        self.train_loader = train_loader
        self.test_loader = test_loader

        # 优化器
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        # 初始化记录器和保存路径
        self.best_test = 0
        self.best_epoch = 0
        self.best_acc = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_f1 = 0
        self.best_auc = 0
        self.best_auprc = 0
        self.log_file = os.path.join(save_path, 'log', 'test_results.csv')
        self.checkpoint_path = os.path.join(save_path, 'checkpoint')

        # 创建保存目录
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        os.makedirs(self.checkpoint_path, exist_ok=True)

        self.edges, self.weights = get_foot_graph_edges_and_weights(left_coords, right_coords)
        self.edges = self.edges.to(device)
        self.weights = self.weights.to(device)

    def train_one_epoch(self):
        self.model.train()  # 将模型设为训练模式
        epoch_loss = 0.0
        correct = 0
        total = 0

        for data, labels, mask in self.train_loader:
            data, labels, mask = data.to(self.device), labels.to(self.device), mask.to(self.device)

            # 清空优化器的梯度
            self.optimizer.zero_grad()

            # 前向传播
            outputs = self.model(data, self.edges, self.weights, mask=mask)

            # 计算损失
            loss = self.criterion(outputs, labels)
            epoch_loss += loss.item()

            # 反向传播
            loss.backward()
            self.optimizer.step()

            # 计算准确度
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        # 计算每轮的训练损失和准确度
        train_acc = 100 * correct / total
        return epoch_loss / len(self.train_loader), train_acc

    def evaluate(self):
        self.model.eval()  # 将模型设为评估模式
        all_labels = []
        all_preds = []
        all_probs = []

        with torch.no_grad():
            for data, labels, mask in self.test_loader:
                data, labels, mask = data.to(self.device), labels.to(self.device), mask.to(self.device)

                # 前向传播
                outputs = self.model(data, self.edges, self.weights, mask=mask)

                # 存储标签和预测结果
                all_labels.extend(labels.cpu().numpy())
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_probs.extend(torch.softmax(outputs, dim=1)[:, 1].cpu().numpy())

        # 计算各种评估指标
        acc = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc = roc_auc_score(all_labels, all_probs)
        auprc = average_precision_score(all_labels, all_probs)

        return acc, precision, recall, f1, auc, auprc

    def save_model(self, epoch):
        # 保存模型参数
        model_checkpoint_path = os.path.join(self.checkpoint_path, f'checkpoint.pt')
        torch.save(self.model.state_dict(), model_checkpoint_path)

    def save_test_results(self, epoch, acc, precision, recall, f1, auc, auprc):
        # 记录测试结果到CSV
        if not os.path.exists(self.log_file):
            df = pd.DataFrame(columns=['epoch', 'acc', 'precision', 'recall', 'f1', 'auc', 'auprc'])
        else:
            df = pd.read_csv(self.log_file)

        # 添加当前epoch的结果
        new_row = {'epoch': epoch, 'acc': acc, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc,
                   'auprc': auprc}
        new_row_df = pd.DataFrame([new_row])
        df = pd.concat([df, new_row_df], ignore_index=True)

        # 保存回CSV
        df.to_csv(self.log_file, index=False)
