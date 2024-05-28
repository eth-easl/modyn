# flake8: noqa
# mypy: ignore-errors

import numpy as np
import torch
from scipy.linalg import lstsq
from scipy.optimize import nnls


def orthogonal_matching_pursuit(A, b, budget: int, lam: float = 1.0):
    """approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Acknowledgement to:
    https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      budget: selection budget
      lam: regularization coef. for the final output vector
    Returns:
       vector of length n
    """
    with torch.no_grad():
        d, n = A.shape
        if budget <= 0:
            budget = 0
        elif budget > n:
            budget = n

        x = np.zeros(n, dtype=np.float32)
        resid = b.clone()
        indices = []
        boolean_mask = torch.ones(n, dtype=bool, device="cuda")
        all_idx = torch.arange(n, device="cuda")

        for i in range(budget):
            projections = torch.matmul(A.T, resid)
            index = torch.argmax(projections[boolean_mask])
            index = all_idx[boolean_mask][index]

            indices.append(index.item())
            boolean_mask[index] = False

            if indices.__len__() == 1:
                A_i = A[:, index]
                x_i = projections[index] / torch.dot(A_i, A_i).view(-1)
                A_i = A[:, index].view(1, -1)
            else:
                A_i = torch.cat((A_i, A[:, index].view(1, -1)), dim=0)
                temp = torch.matmul(A_i, torch.transpose(A_i, 0, 1)) + lam * torch.eye(A_i.shape[0], device="cuda:0")
                lstsq_out = torch.linalg.lstsq(temp, torch.matmul(A_i, b).view(-1, 1))
                x_i = torch.cat((lstsq_out.solution, lstsq_out.residuals))
            resid = b - torch.matmul(torch.transpose(A_i, 0, 1), x_i).view(-1)
        if budget > 1:
            x_i = nnls(temp.cpu().numpy(), torch.matmul(A_i, b).view(-1).cpu().numpy())[0]
            x[indices] = x_i
        elif budget == 1:
            x[indices[0]] = 1.0
    return x


def orthogonal_matching_pursuit_np(A, b, budget: int, lam: float = 1.0):
    """approximately solves min_x |x|_0 s.t. Ax=b using Orthogonal Matching Pursuit
    Acknowledgement to:
    https://github.com/krishnatejakk/GradMatch/blob/main/GradMatch/selectionstrategies/helpers/omp_solvers.py
    Args:
      A: design matrix of size (d, n)
      b: measurement vector of length d
      budget: selection budget
      lam: regularization coef. for the final output vector
    Returns:
       vector of length n
    """
    d, n = A.shape
    if budget <= 0:
        budget = 0
    elif budget > n:
        budget = n

    x = np.zeros(n, dtype=np.float32)
    resid = np.copy(b)
    indices = []
    boolean_mask = np.ones(n, dtype=bool)
    all_idx = np.arange(n)

    for i in range(budget):
        projections = A.T.dot(resid)
        index = np.argmax(projections[boolean_mask])
        index = all_idx[boolean_mask][index]

        indices.append(index.item())
        boolean_mask[index] = False

        if indices.__len__() == 1:
            A_i = A[:, index]
            x_i = projections[index] / A_i.T.dot(A_i)
        else:
            A_i = np.vstack([A_i, A[:, index]])
            x_i = lstsq(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
        resid = b - A_i.T.dot(x_i)
    if budget > 1:
        x_i = nnls(A_i.dot(A_i.T) + lam * np.identity(A_i.shape[0]), A_i.dot(b))[0]
        x[indices] = x_i
    elif budget == 1:
        x[indices[0]] = 1.0
    return x
