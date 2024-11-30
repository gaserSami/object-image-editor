import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse.linalg import spsolve

def compute_backward_differences(img):
  """
  Compute backward differences for an image.

  Parameters:
  img (numpy.ndarray): Input image.

  Returns:
  dict: Dictionary containing backward differences in x and y directions.
  """
  D_X = np.zeros_like(img)
  D_X[:, 1:, :] = img[:, 1:, :] - img[:, :-1, :]
  D_Y = np.zeros_like(img)
  D_Y[1:, :, :] = img[1:, :, :] - img[:-1, :, :] 
  
  return {'x': D_X, 'y': D_Y}

def compute_mixed_DV(DG, DF, O):
  """
  Compute mixed gradient vector field.

  Parameters:
  DG (dict): Gradient of the source image.
  DF (dict): Gradient of the destination image.
  O (numpy.ndarray): Mask.

  Returns:
  dict: Dictionary containing mixed gradient vector field in x and y directions.
  """
  V_X = DF["x"].copy()
  V_Y = DF["y"].copy()

  O = np.pad(O, pad_width=1, mode="constant", constant_values=False)
  O = np.roll(O, -1, axis=0) | np.roll(O, 1, axis=0) | np.roll(O, -1, axis=1) | np.roll(O, 1, axis=1)
  O = O[1:-1, 1:-1]

  V_X[O] = np.where(np.abs(DF["x"][O]) > np.abs(DG["x"][O]), DF["x"][O], DG["x"][O])
  V_Y[O] = np.where(np.abs(DF["y"][O]) > np.abs(DG["y"][O]), DF["y"][O], DG["y"][O])
  return {'x': V_X, 'y': V_Y}

def solve_poisson(V, O, F):
  """
  Solve the Poisson equation for each channel.

  Parameters:
  V (dict): Mixed gradient vector field.
  O (numpy.ndarray): Mask.
  F (numpy.ndarray): Destination image.

  Returns:
  numpy.ndarray: Resulting image after Poisson editing.
  """
  I = np.zeros_like(F)
  for c in range(F.shape[2]):
    I[:,:,c] = solve_possion_channel(V['x'][:,:,c], V['y'][:,:,c], O, F[:,:,c])
  return I

def solve_possion_channel(V_X, V_Y, O, F):
  """
  Solve the Poisson equation for a single channel.

  Parameters:
  V_X (numpy.ndarray): Gradient in x direction.
  V_Y (numpy.ndarray): Gradient in y direction.
  O (numpy.ndarray): Mask.
  F (numpy.ndarray): Destination image channel.

  Returns:
  numpy.ndarray: Resulting channel after Poisson editing.
  """
  PW = 1
  F = F.astype(np.float64)
  V_X = np.pad(V_X, pad_width=PW, mode="constant", constant_values=0)
  V_Y = np.pad(V_Y, pad_width=PW, mode="constant", constant_values=0)
  O = np.pad(O, pad_width=PW, mode="symmetric")
  F = np.pad(F, pad_width=PW, mode="symmetric")

  H, W = F.shape
  HW = H * W

  N = (H+2) * (W+2)
  mask = np.zeros((H+2, W+2))
  mask[1:-1, 1:-1] = 1
  idxs = np.flatnonzero(mask.T)

  D_O = np.pad(O, pad_width=PW, mode="constant", constant_values=False)
  D_O = D_O | np.roll(D_O, 1, axis=0) | np.roll(D_O, -1, axis=0) | np.roll(D_O, 1, axis=1) | np.roll(D_O, -1, axis=1)
  D_O[[0, -1], :] = False
  D_O[:, [0, -1]] = False
  idx = np.flatnonzero((D_O.astype(np.int32)).T)
  
  Dx = (sparse.csr_matrix((np.ones(len(idx)), (idx, idx+(H+2))), shape=(N, N)) - 
      sparse.csr_matrix((np.ones(len(idx)), (idx, idx)), shape=(N, N)))
  
  Dy = (sparse.csr_matrix((np.ones(len(idx)), (idx, idx+1)), shape=(N,N)) - 
      sparse.csr_matrix((np.ones(len(idx)), (idx, idx)), shape=(N,N)))
  
  L = (sparse.csr_matrix((np.full(len(idx), -4), (idx, idx)), shape=(N,N)) +
     sparse.csr_matrix((np.ones(len(idx)), (idx, idx+1)), shape=(N,N)) +
     sparse.csr_matrix((np.ones(len(idx)), (idx, idx-1)), shape=(N,N)) +
     sparse.csr_matrix((np.ones(len(idx)), (idx, idx+(H+2))), shape=(N,N)) +
     sparse.csr_matrix((np.ones(len(idx)), (idx, idx-(H+2))), shape=(N,N)))
  
  Dx = Dx[np.ix_(idxs, idxs)]
  Dy = Dy[np.ix_(idxs, idxs)]
  L = L[np.ix_(idxs, idxs)]
  
  Dx = Dx - sparse.csr_matrix((Dx.sum(axis=1).A1, (range(HW), range(HW))), shape=(HW,HW))
  Dy = Dy - sparse.csr_matrix((Dy.sum(axis=1).A1, (range(HW), range(HW))), shape=(HW,HW))
  L = L - sparse.csr_matrix((L.sum(axis=1).A1, (range(HW), range(HW))), shape=(HW,HW))

  M_O = sparse.diags((O.astype(np.int32)).flatten("F"))
  
  D_O = np.pad(O, pad_width=1, mode='constant', constant_values=False)
  D_O = (np.roll(D_O, 1, axis=0) | np.roll(D_O, -1, axis=0) |
        np.roll(D_O, 1, axis=1) | np.roll(D_O, -1, axis=1))
  D_O = D_O[1:-1, 1:-1]
  D_O[O == True] = False

  M_D_O = sparse.diags(D_O.astype(np.int32).flatten("F"))

  idx = np.flatnonzero((O.astype(np.int32)).T)
  S = sparse.csr_matrix((np.ones(len(idx)), (range(len(idx)), idx)), 
            shape=(len(idx), HW))
  
  A = L @ M_O
  A = S @ A @ S.T
  b = (Dx @ V_X.flatten("F")) + (Dy @ V_Y.flatten("F")) - (L @ M_D_O @ F.flatten("F"))
  b = S @ b
  x = spsolve(A, b)

  I = F.copy()
  i = 0
  
  # TODO: Vectorize this
  for col in range(F.shape[1]):
    for row in range(F.shape[0]):
      if O[row, col]:
        I[row, col] = x[i]
        i += 1

  I = I[1:-1, 1:-1]

  return I

def poisson_edit(G, F, O):
  """
  Perform Poisson image editing.

  Parameters:
  G (numpy.ndarray): Source image.
  F (numpy.ndarray): Destination image.
  O (numpy.ndarray): Mask.

  Returns:
  numpy.ndarray: Resulting image after Poisson editing.
  """
  # Pre-process
  O[O < 128] = 0
  O[O >= 128] = 1
  O = O.astype(bool)
  F = F.astype(np.float64)
  G = G.astype(np.float64)
  
  # Main process
  DG = compute_backward_differences(G)
  DF = compute_backward_differences(F)
  V = compute_mixed_DV(DG, DF, O.astype(bool))
  I = solve_poisson(V, O, F)

  return I