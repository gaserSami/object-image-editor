import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def _compute_backward_diffs(img):
  """
  Compute backward differences for an image.

  Parameters:
  img (numpy.ndarray): Input image.

  Returns:
  dict: Dictionary containing backward differences in x and y directions.
  """
  DX = np.zeros_like(img)
  DX[:, 1:, :] = img[:, 1:, :] - img[:, :-1, :]
  DY = np.zeros_like(img)
  DY[1:, :, :] = img[1:, :, :] - img[:-1, :, :] 
  
  return {'x': DX, 'y': DY}

def _compute_mixed_dir_vecs(source_diff, target_diff, omega, mode="Max"): # Max, Sum, Average, Min, Replace
  """
  Compute mixed gradient vector field.

  Parameters:
  source_diff (dict): Gradient of the source image.
  target_diff (dict): Gradient of the destination image.
  omega (numpy.ndarray): Mask.

  Returns:
  dict: Dictionary containing mixed gradient vector field in x and y directions.
  """
  dir_vec_x = target_diff["x"].copy()
  dir_vec_y = target_diff["y"].copy()

  omega = np.pad(omega, pad_width=1, mode="constant", constant_values=False)
  omega = np.roll(omega, -1, axis=0) | np.roll(omega, 1, axis=0) | np.roll(omega, -1, axis=1) | np.roll(omega, 1, axis=1)
  omega = omega[1:-1, 1:-1]

  if mode == "Replace":
        dir_vec_x[omega] = source_diff["x"][omega]
        dir_vec_y[omega] = source_diff["y"][omega]
  elif mode == "Average":
      dir_vec_x[omega] = 0.5 * (source_diff["x"][omega] + target_diff["x"][omega])
      dir_vec_y[omega] = 0.5 * (source_diff["y"][omega] + target_diff["y"][omega])
  elif mode == "Sum":
      dir_vec_x[omega] = source_diff["x"][omega] + target_diff["x"][omega]
      dir_vec_y[omega] = source_diff["y"][omega] + target_diff["y"][omega]
  elif mode == "Max":
      dir_vec_x[omega] = np.where(np.abs(target_diff["x"][omega]) > np.abs(source_diff["x"][omega]), target_diff["x"][omega], source_diff["x"][omega])
      dir_vec_y[omega] = np.where(np.abs(target_diff["y"][omega]) > np.abs(source_diff["y"][omega]), target_diff["y"][omega], source_diff["y"][omega])
  else:
      raise ValueError("Mode Unknown")
  
  return {'x': dir_vec_x, 'y': dir_vec_y}

def _solve_poisson(dir_vec, omega, target_img):
  """
  Solve the Poisson equation for each channel.

  Parameters:
  dir_vec (dict): Mixed gradient vector field.
  omega (numpy.ndarray): Mask.
  target_img (numpy.ndarray): Destination image.

  Returns:
  numpy.ndarray: Resulting image after Poisson editing.
  """
  img = np.zeros_like(target_img)
  for c in range(target_img.shape[2]):
    img[:,:,c] = _solve_possion_channel(dir_vec['x'][:,:,c], dir_vec['y'][:,:,c], omega, target_img[:,:,c])
  return img

def _solve_possion_channel(dir_vec_x, dir_vec_y, omega, target_img):
  """
  Solve the Poisson equation for a single channel.

  Parameters:
  dir_vec_x (numpy.ndarray): Gradient in x direction.
  dir_vec_y (numpy.ndarray): Gradient in y direction.
  omega (numpy.ndarray): Mask.
  target_img (numpy.ndarray): Destination image channel.

  Returns:
  numpy.ndarray: Resulting channel after Poisson editing.
  """
  PW = 1
  target_img = target_img.astype(np.float64)
  dir_vec_x = np.pad(dir_vec_x, pad_width=PW, mode="constant", constant_values=0)
  dir_vec_y = np.pad(dir_vec_y, pad_width=PW, mode="constant", constant_values=0)
  omega = np.pad(omega, pad_width=PW, mode="symmetric")
  target_img = np.pad(target_img, pad_width=PW, mode="symmetric")

  H, W = target_img.shape
  HW = H * W

  N = (H+2) * (W+2)
  mask = np.zeros((H+2, W+2))
  mask[1:-1, 1:-1] = 1
  idxs = np.flatnonzero(mask.T)

  d_omega = np.pad(omega, pad_width=PW, mode="constant", constant_values=False)
  d_omega = d_omega | np.roll(d_omega, 1, axis=0) | np.roll(d_omega, -1, axis=0) | np.roll(d_omega, 1, axis=1) | np.roll(d_omega, -1, axis=1)
  d_omega[[0, -1], :] = False
  d_omega[:, [0, -1]] = False
  idx = np.flatnonzero((d_omega.astype(np.int32)).T)
  
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

  omega_proj_mat = sparse.diags((omega.astype(np.int32)).flatten("F"))
  
  d_omega = np.pad(omega, pad_width=1, mode='constant', constant_values=False)
  d_omega = (np.roll(d_omega, 1, axis=0) | np.roll(d_omega, -1, axis=0) |
        np.roll(d_omega, 1, axis=1) | np.roll(d_omega, -1, axis=1))
  d_omega = d_omega[1:-1, 1:-1]
  d_omega[omega == True] = False

  d_omega_proj_mat = sparse.diags(d_omega.astype(np.int32).flatten("F"))

  idx = np.flatnonzero((omega.astype(np.int32)).T)
  sampling_mat = sparse.csr_matrix((np.ones(len(idx)), (range(len(idx)), idx)), 
            shape=(len(idx), HW))
  
  A = L @ omega_proj_mat
  A = sampling_mat @ A @ sampling_mat.T
  b = (Dx @ dir_vec_x.flatten("F")) + (Dy @ dir_vec_y.flatten("F")) - (L @ d_omega_proj_mat @ target_img.flatten("F"))
  b = sampling_mat @ b
  x = spsolve(A, b)

  img = target_img.copy()
  i = 0
  
  # TODO: Vectorize this
  for col in range(target_img.shape[1]):
    for row in range(target_img.shape[0]):
      if omega[row, col]:
        img[row, col] = x[i]
        i += 1

  img = img[1:-1, 1:-1]

  return img

def poisson_edit(source_img, target_img, omega, mode):
  """
  Perform Poisson image editing.

  Parameters:
  source_img (numpy.ndarray): Source image.
  target_img (numpy.ndarray): Destination image.
  omega (numpy.ndarray): Mask.

  Returns:
  numpy.ndarray: Resulting image after Poisson editing.
  """
  # Pre-process
  omega[omega < 128] = 0
  omega[omega >= 128] = 1
  omega = omega.astype(bool)
  target_img = target_img.astype(np.float64)
  source_img = source_img.astype(np.float64)
  
  # Main process
  source_diff = _compute_backward_diffs(source_img)
  target_diff = _compute_backward_diffs(target_img)
  dir_vec = _compute_mixed_dir_vecs(source_diff, target_diff, omega.astype(bool), mode)
  img = _solve_poisson(dir_vec, omega, target_img)

  return img