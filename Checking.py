import numpy as np, torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader, Subset
from laplace import Laplace
import matplotlib.pyplot as plt
import laplace
import pickle
import os
import random
import time

from utils.DataUtils import save_datasets, load_datasets, extract_samples_from_unlabeled, delete_samples_from_unlabeled, add_samples_to_labeled, add_and_extract_and_delete_samples
from utils.ModelUtils import initialize_model_weights, train_model, evaluate, count_parameters, save_model, load_model
from utils.LaplaceUtils import return_hessian_eigenvalues, compute_outcome_hessian_from_model, symmetric_matrix_sqrt, fast_jacobian, low_rank_updated_part
from utils.AlFunctions import DoptScore_per_sample, AoptScore_per_sample, ToptScore_per_sample, selection_AL_scores, AL_finetune_model

