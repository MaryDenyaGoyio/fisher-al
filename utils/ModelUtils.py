'''Libraries'''
import numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, Subset

import pickle
import os
import random
import time

def initialize_model_weights(model, init_type='xavier'):
    """
    모델 가중치를 초기화합니다.
    init_type: 'xavier', 'kaiming', 'normal', 'uniform'
    """
    for m in model.modules():
        if isinstance(m, nn.Linear):
            if init_type == 'xavier':
                nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'normal':
                nn.init.normal_(m.weight, mean=0, std=0.01)
            elif init_type == 'uniform':
                nn.init.uniform_(m.weight, -0.1, 0.1)
            
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model

def get_models():
    models = []

    # 1. 가장 단순한 모델
    models.append(nn.Linear(64, 10))

    # 2. 파라미터 2배: Linear + Linear
    models.append(nn.Sequential(
        nn.Linear(64, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ))

    # 3. 더 깊게: Linear + Linear + Linear
    models.append(nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    ))

    # 4. 더 깊고 넓게
    models.append(nn.Sequential(
        nn.Linear(64, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ))

    # 5. BatchNorm 추가
    models.append(nn.Sequential(
        nn.Linear(64, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.Linear(256, 10)
    ))

    # 6. Dropout 추가
    models.append(nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ))

    # 7. 더 깊게, 더 넓게
    models.append(nn.Sequential(
        nn.Linear(64, 256),
        nn.ReLU(),
        nn.Linear(256, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ))

    # 8. 더 많은 레이어와 BatchNorm, Dropout
    models.append(nn.Sequential(
        nn.Linear(64, 512),
        nn.BatchNorm1d(512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ))

    # 9. 더 깊고 넓게, 활성화 다양화
    models.append(nn.Sequential(
        nn.Linear(64, 512),
        nn.ReLU(),
        nn.Linear(512, 512),
        nn.Tanh(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ))

    # 10. 가장 큰 모델
    models.append(nn.Sequential(
        nn.Linear(64, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 10)
    ))

    return models

def train_model(model, train_set_Labeled, criterion, optimizer, n_epochs=5):
    n_epochs_monitor = n_epochs
    for epoch in range(n_epochs_monitor):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_set_Labeled:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    return model

def evaluate(loader, model):
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return correct / total

def count_parameters(model, only_trainable=False):
    """
    모델의 파라미터 개수를 반환합니다.
    only_trainable=True이면 requires_grad=True인 파라미터만 셉니다.
    """
    params = (p for p in model.parameters() if (not only_trainable) or p.requires_grad)
    return sum(p.numel() for p in params)

def save_model(model, save_path='checkpoints/model.pth'):
    """
    모델의 state_dict를 저장
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"✅ Saved model to {save_path}")

def load_model(model, load_path='checkpoints/model.pth'):
    """
    저장된 state_dict를 모델에 로드
    
    Args:
        model: 빈 모델 (구조만 정의된 상태)
        load_path: 저장된 모델 경로
    
    Returns:
        model: 가중치가 로드된 모델
    """
    model.load_state_dict(torch.load(load_path))
    print(f"✅ Loaded model from {load_path}")
    return model
