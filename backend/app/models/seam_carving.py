import cv2
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import os

# temp for debugging
def make_gif(intermediate_images, image):
    # Create tmp directory if it doesn't exist
        os.makedirs('tmp', exist_ok=True)
        
        # Generate unique filename
        gif_path = os.path.join('tmp', f'seam_carving.gif')
        
        # Convert images to uint8 and pad smaller images to match the original size
        uint8_images = []
        original_shape = image.shape
        for img in intermediate_images:
            # Pad the image to match original dimensions
            padded_img = np.zeros(original_shape, dtype=np.uint8)
            h, w = img.shape[:2]
            padded_img[:h, :w] = img.astype(np.uint8)
            uint8_images.append(padded_img)
            
        imageio.mimsave(gif_path, uint8_images, duration=0.1)  # 0.1 seconds per frame
        return gif_path


########################################
# CONSTANTS
########################################

FACTOR = 1e5

########################################
# CORE SEAM OPERATIONS
########################################

def remove_seams(img, num_seams=0, protect_mask=None, remove_mask=None, forward=True):
    """Main function to remove seams either by count or based on removal mask"""
    if img is None:
        return img, [], []
    try:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
        removed_seams = []
        intermediate_images = [img.copy()]
        new_img = img.copy()
        curr_protect_mask = protect_mask.copy() if protect_mask is not None else None
        curr_remove_mask = remove_mask.copy() if remove_mask is not None else None
        
        # Determine number of seams to remove
        if num_seams == 0:
            # save the curr_remove_mask before removing seams to tmp/ as image
            while np.any(curr_remove_mask):
                cv2.imwrite('tmp/remove_mask.png', (curr_remove_mask * 255).astype(np.uint8))
                new_img, gray, curr_remove_mask, curr_protect_mask, seam = _remove_single_seam(
                    new_img, gray, curr_protect_mask, curr_remove_mask, forward)
                removed_seams.append(seam)
                intermediate_images.append(curr_remove_mask.copy())
        else:
            for _ in range(num_seams):
                new_img, gray, curr_remove_mask, curr_protect_mask, seam = _remove_single_seam(
                new_img, gray, curr_protect_mask, curr_remove_mask, forward)
                removed_seams.append(seam)
                intermediate_images.append(new_img.copy())
    except:
        gif_path = make_gif(intermediate_images=intermediate_images, image=img)
        print(gif_path)
        return None, [], intermediate_images

    return new_img, removed_seams, intermediate_images

def _remove_single_seam(img, gray, protect_mask, remove_mask, forward):
    """Helper function to remove a single seam"""
    cum_energyy, backtrack_path = cum_energy(gray, protect_mask=protect_mask, 
                                           remove_mask=remove_mask, forward=forward)
    boolean_path, coords_path = getMinPathMask(backtrack_path, cum_energyy)
    new_img, new_remove_mask, new_protect_mask, _ = remove_path(
        img, remove_mask=remove_mask, protection_mask=protect_mask,
        boolean_path=boolean_path, path=coords_path)
    new_gray = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return new_img, new_gray, new_remove_mask, new_protect_mask, coords_path

def add_seams(img, num_seams=1, protect_mask=None, forward=True):
    """Main function to add seams to an image"""
    new_img = img.copy()
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    new_mask = protect_mask.copy() if protect_mask is not None else None

    # Store seams and their insertion positions
    tmp_img = img.copy()
    tmp_mask = protect_mask.copy() if protect_mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    H,W = tmp_img.shape[:2]
    removed_seams = []
    idx_map = np.tile(np.arange(W), (H, 1))
    tmp_idx_map = idx_map.copy()
    # Find and store all seams first
    for _ in range(num_seams):
        cum_energyy, backtrack_path = cum_energy(gray, protect_mask=tmp_mask, forward=forward)
        boolean_path, coords_path = getMinPathMask(backtrack_path, cum_energyy)
        tmp_img, _, tmp_mask, tmp_idx_map = remove_path(tmp_img, 
                                            remove_mask=None,
                                            protection_mask=tmp_mask, 
                                            boolean_path=boolean_path,
                                            path=coords_path,
                                            idx_map=tmp_idx_map
                                            )
        updated_path = coords_path.copy()
        updated_path[:, 1] = idx_map[coords_path[:, 0], coords_path[:, 1]]
        gray = cv2.cvtColor(tmp_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
        removed_seams.append(updated_path)

    # Insert the seams
    for seam in removed_seams:
        new_img, new_mask = insert_seam(new_img, seam, protection_mask=new_mask)
        
    return new_img, new_mask

########################################
# ENERGY COMPUTATION
########################################

def compute_energy(img):
    """Computes basic gradient energy"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy

def cum_energy(img, protect_mask=None, remove_mask=None, forward=True):
    """Computes cumulative energy matrix"""
    if not forward:
        return _backward_energy(img, protect_mask, remove_mask)
    return _forward_energy(img, protect_mask, remove_mask)

def _backward_energy(img, protect_mask, remove_mask):
    """Helper function for backward energy computation"""
    energy = compute_energy(img)
    # normalize energy by dividing by max
    energy = energy / energy.max()
    H, W = energy.shape
    M = energy.copy()
    backtrack_path = np.zeros((H, W), dtype=np.int32)

    if protect_mask is None:
        protect_mask = np.zeros((H, W), dtype=np.float64)
    if remove_mask is None:
        remove_mask = np.zeros((H, W), dtype=np.float64)

    protect_mask = protect_mask.astype(np.float64)
    remove_mask = remove_mask.astype(np.float64)

    protect_mask *= FACTOR
    remove_mask *= -FACTOR

    M += protect_mask + remove_mask
    
    for i in range(1, H):
        prev_row = np.pad(M[i - 1], (1, 1), mode='constant', constant_values=np.inf)
        left = prev_row[:-2]
        center = prev_row[1:-1]
        right = prev_row[2:]
        lcr = np.array([left, center, right])
        min_choices = np.argmin(lcr, axis=0) # 0 = left, 1 = center, 2 = right
        backtrack_path[i] = min_choices - 1
        M[i] += np.choose(min_choices, lcr)

    return M, backtrack_path

def _forward_energy(img, protect_mask, remove_mask):
    """Helper function for forward energy computation"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray_padded = np.pad(gray, pad_width=1, mode='edge')
    H, W = gray.shape
    backtrack_path = np.zeros((H, W), dtype=np.int32)
    curr_r = gray_padded[0, 2:]
    curr_l = gray_padded[0, :-2]
    CU = np.abs(curr_r - curr_l)
    M = np.zeros((H, W), dtype=np.float64)
    M[0] = CU

    if protect_mask is None:
        protect_mask = np.zeros((H, W), dtype=np.float64)
    if remove_mask is None:
        remove_mask = np.zeros((H, W), dtype=np.float64)

    protect_mask = protect_mask.astype(np.float64)
    remove_mask = remove_mask.astype(np.float64)

    protect_mask *= FACTOR
    remove_mask *= -FACTOR

    M[0] = M[0] + protect_mask[0] + remove_mask[0]

    for i in range(1, H):
        curr_r = gray_padded[i, 2:]
        curr_l = gray_padded[i, :-2]
        CU = np.abs(curr_r - curr_l)
        prev_row_U = gray[i - 1]
        CL = np.abs(prev_row_U - curr_l) + CU
        CR = np.abs(prev_row_U - curr_r) + CU
        prev_row = np.pad(M[i - 1], (1, 1), mode='constant', constant_values=np.inf)
        left = prev_row[:-2]
        center = prev_row[1:-1]
        right = prev_row[2:]
        lcr = np.array([left, center, right])
        lcr += np.array([CL, CU, CR])
        min_choices = np.argmin(lcr, axis=0) # 0 = left, 1 = center, 2 = right
        backtrack_path[i] = min_choices - 1
        M[i] += np.choose(min_choices, lcr) + protect_mask[i] + remove_mask[i]

    return M, backtrack_path

########################################
# PATH OPERATIONS
########################################

def remove_path(img, path, boolean_path=None, remove_mask=None, protection_mask=None, idx_map=None):
    """Removes a seam path from image and associated masks"""
    H, W, C = img.shape 
    new_img = np.zeros((H, W - 1, C), dtype=np.float64)
    new_remove_mask = None 
    new_protection_mask = None
    new_idx_map = None
    
    if remove_mask is not None:
        new_remove_mask = np.zeros((H, W - 1), dtype=np.float64)
    if protection_mask is not None:
        new_protection_mask = np.zeros((H, W - 1), dtype=np.float64)
    if idx_map is not None:
        new_idx_map = np.zeros((H, W - 1), dtype=np.int32)

    for y, x in path:
        new_img[y, :x, :] = img[y, :x, :]
        new_img[y, x:, :] = img[y, x+1:, :]
        if remove_mask is not None:
            new_remove_mask[y, :x] = remove_mask[y, :x]
            new_remove_mask[y, x:] = remove_mask[y, x+1:]
        if protection_mask is not None:
            new_protection_mask[y, :x] = protection_mask[y, :x]
            new_protection_mask[y, x:] = protection_mask[y, x+1:]
        if idx_map is not None:
            new_idx_map[y, :x] = idx_map[y, :x]
            new_idx_map[y, x:] = idx_map[y, x+1:]

    return new_img, new_remove_mask, new_protection_mask, new_idx_map

def insert_seam(img, path, boolean_path=None, protection_mask=None):
    """Inserts a seam path into image"""
    H, W, C = img.shape
    new_img = np.zeros((H, W + 1, C), dtype=img.dtype)
    new_protection_mask = None

    if protection_mask is not None:
        new_protection_mask = np.zeros((H, W + 1), dtype=np.uint8)

    for y, x in path:
        if x == 0:
            new_img[y,x,:] = img[y,x,:]
            new_val = ((img[y,x,:].astype(np.float32) + img[y,x+1,:].astype(np.float32)) / 2).astype(np.uint8)
            new_img[y,x+1,:] = new_val
            new_img[y,x+1:,:] = img[y,x:,:]
            if protection_mask is not None:
                new_protection_mask[y,x] = protection_mask[y,x]
                new_protection_mask[y,x+1] = max(protection_mask[y,x], protection_mask[y,x+1])
                new_protection_mask[y,x+1:] = protection_mask[y,x:]
        else:
            new_img[y, :x, :] = img[y, :x, :]
            new_val = ((img[y, x, :].astype(np.float32) + img[y, x - 1, :].astype(np.float32)) / 2).astype(np.uint8)
            new_img[y, x, :] = new_val
            new_img[y, x + 1:, :] = img[y, x:, :]
            if protection_mask is not None:
                new_protection_mask[y, :x] = protection_mask[y, :x]
                new_protection_mask[y, x] = max(protection_mask[y, x], protection_mask[y, x - 1])
                new_protection_mask[y, x + 1:] = protection_mask[y, x:]

    return new_img, new_protection_mask

def getMinPathMask(backtrack_path, cum_energy):
    """Finds minimum energy seam path"""
    H, W = backtrack_path.shape
    path = np.empty((H,2), dtype=np.int64)
    x = cum_energy[-1].argmin()
    mask = np.zeros((H, W), np.uint8)

    for i in range(H - 1, -1, -1):
        path[i] = (i, x)
        mask[i, x] = 1
        x = x + backtrack_path[i, x]
        x = max(0, min(W - 1, x))

    return mask, path # boolean_path, coords_path

########################################
# UTILITY FUNCTIONS
########################################

def get_masked_obj_dimensions(mask):
    """Gets dimensions of masked object"""
    if mask is None:
        return 0, 0
    mask = np.asarray(mask)
    
    y_indices, x_indices = np.where(mask)
    
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0
    else:
        width = x_indices.max() - x_indices.min() + 1
        height = y_indices.max() - y_indices.min() + 1
        return width, height

def draw_seam(img, coords_path):
    """Debug function to visualize seam"""
    seam = np.copy(img)
    for y,x in coords_path:
            seam[y, x] = 255
    return seam

########################################
# PUBLIC INTERFACE
########################################

def compute_rotation_decision(img, remove_mask):
    """Determine if rotation is needed based on gradient energy analysis"""
    # Calculate gradients in both directions
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    
    # Calculate average energy in both directions within the mask
    mask_3d = np.stack([remove_mask] * 3, axis=-1)
    horizontal_energy = np.mean(np.abs(grad_x)[mask_3d == 1])
    vertical_energy = np.mean(np.abs(grad_y)[mask_3d == 1])
    
    # Get dimensions of the object
    w, h = get_masked_obj_dimensions(remove_mask)
    
    # Consider both energy and dimensions in decision
    energy_ratio = horizontal_energy / vertical_energy
    dimension_ratio = w / h
    
    # Rotate if vertical energy is significantly lower and width is larger
    # or if horizontal energy is much stronger despite dimensions
    return energy_ratio > 1.2 or (dimension_ratio > 1.5 and energy_ratio > 0.8)

def remove(mainImg, remove_mask, protect_mask=None, forward=True, direction="auto"):
    """
    Main public interface for object removal
    
    Parameters:
        mainImg: Input image
        remove_mask: Mask of area to remove
        protect_mask: Mask of area to protect (optional)
        forward: Whether to use forward energy (default: True)
        direction: Forced direction for seam removal ('vertical', 'horizontal', or "auto" for auto)
    """
    if direction != "auto":
        # Use specified direction
        should_rotate = direction.lower() == 'horizontal'
    else:
        # Use automatic direction detection
        should_rotate = compute_rotation_decision(mainImg, remove_mask)
    
    if should_rotate:
        mainImg = np.rot90(mainImg)
        remove_mask = np.rot90(remove_mask)
        if protect_mask is not None:
            protect_mask = np.rot90(protect_mask)
    
    removed_img, removed_seams, intermediate_images = remove_seams(
        img=mainImg, remove_mask=remove_mask, 
        protect_mask=protect_mask, forward=forward)

    if should_rotate:
        removed_img = np.rot90(removed_img, -1)
        intermediate_images = [np.rot90(img, -1) for img in intermediate_images]

    return removed_img, removed_seams, intermediate_images