import laplace
import torch
from laplace import Laplace
from torch.utils.data import TensorDataset, DataLoader, Subset

def return_hessian_eigenvalues(model, train_set_Labeled):
    la = Laplace(model, 'classification', subset_of_weights='all', hessian_structure='diag')
    train_loader = DataLoader(train_set_Labeled, batch_size=32, shuffle=True)
    la.fit(train_loader)
    val = la.H
    return val