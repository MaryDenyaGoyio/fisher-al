import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import pickle
import os

def save_datasets(train_set_Labeled, train_set_Unlabeled, test_set, save_path='checkpoints/datasets.pkl'):
    """
    현재 train_set_Labeled, train_set_Unlabeled, test_set을 저장
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    data = {
        'labeled_indices': train_set_Labeled.indices,
        'unlabeled_indices': train_set_Unlabeled.indices,
        'full_dataset': train_set_Labeled.dataset,  # 전체 dataset (동일함)
        'test_set': test_set
    }
    
    with open(save_path, 'wb') as f:
        pickle.dump(data, f)
    
    print(f"✅ Saved datasets to {save_path}")
    print(f"   - Labeled: {len(train_set_Labeled)} samples")
    print(f"   - Unlabeled: {len(train_set_Unlabeled)} samples")
    print(f"   - Test: {len(test_set)} samples")

def load_datasets(load_path='checkpoints/datasets.pkl'):
    """
    저장된 dataset들을 불러오기
    
    Returns:
        train_set_Labeled, train_set_Unlabeled, test_set
    """
    with open(load_path, 'rb') as f:
        data = pickle.load(f)
    
    full_dataset = data['full_dataset']
    train_set_Labeled = Subset(full_dataset, data['labeled_indices'])
    train_set_Unlabeled = Subset(full_dataset, data['unlabeled_indices'])
    test_set = data['test_set']
    
    print(f"✅ Loaded datasets from {load_path}")
    print(f"   - Labeled: {len(train_set_Labeled)} samples")
    print(f"   - Unlabeled: {len(train_set_Unlabeled)} samples")
    print(f"   - Test: {len(test_set)} samples")
    
    return train_set_Labeled, train_set_Unlabeled, test_set

def extract_samples_from_unlabeled(unlabeled_set, selected_indices):
    """
    Extract samples at selected_indices from unlabeled_set.
    Returns: list of (input, label) tuples.
    """
    samples = [unlabeled_set[i] for i in selected_indices]
    return samples

def delete_samples_from_unlabeled(unlabeled_set, selected_indices):
    """
    Remove samples at selected_indices from unlabeled_set.
    Returns: new Subset of unlabeled_set without selected samples.
    """
    all_indices = list(range(len(unlabeled_set)))
    remaining_indices = [i for j, i in enumerate(all_indices) if j not in selected_indices]
    new_unlabeled_set = Subset(unlabeled_set.dataset, [unlabeled_set.indices[i] for i in remaining_indices])
    return new_unlabeled_set

def add_samples_to_labeled(labeled_set, unlabeled_set, selected_indices):
    """
    Add samples at selected_indices from unlabeled_set to labeled_set.
    Returns: new Subset of labeled_set with added samples.
    """
    new_indices = labeled_set.indices + [unlabeled_set.indices[i] for i in selected_indices]
    new_labeled_set = Subset(labeled_set.dataset, new_indices)
    return new_labeled_set

def add_and_extract_and_delete_samples(labeled_set, unlabeled_set, selected_indices):
    new_labeled_set = add_samples_to_labeled(labeled_set, unlabeled_set, selected_indices)
    extracted_samples = extract_samples_from_unlabeled(unlabeled_set, selected_indices)
    new_unlabeled_set = delete_samples_from_unlabeled(unlabeled_set, selected_indices)
    return new_labeled_set, new_unlabeled_set, extracted_samples