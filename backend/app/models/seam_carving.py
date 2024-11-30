import cv2
import numpy as np
import matplotlib.pyplot as plt

FACTOR = 1200

########################################
# UTILITY FUNCTIONS
########################################

# this is for debugging purposes
def draw_seam(img, coords_path):
    seam = np.copy(img)
    for y,x in coords_path:
            seam[y, x] = 255
    return seam

def get_masked_obj_dimensions(mask):
    """
    Gets the width and height of the object defined by a binary mask.
    
    Args:
        mask (numpy.ndarray): Binary mask array
        
    Returns:
        tuple: (width, height) of the masked object
    """
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

########################################
# ENERGY FUNCTIONS
########################################

def compute_energy(img):
    """
    Computes the energy of the image using gradient magnitude (e1 energy).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    energy = np.abs(grad_x) + np.abs(grad_y)
    return energy

def cum_energy(img, protect_mask=None, remove_mask=None):
    """
    Computes cumulative energy matrix using forward energy.
    
    Args:
        img (numpy.ndarray): Input image
        protect_mask (numpy.ndarray, optional): Mask of protected pixels
        remove_mask (numpy.ndarray, optional): Mask of pixels to remove
        
    Returns:
        tuple: (cumulative energy matrix, backtracking path matrix)
    """
    energy = compute_energy(img) 
    H, W = energy.shape
    M = energy.copy()
    backtrack_path = np.zeros((H, W), dtype=np.int32)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    if protect_mask is None:
        protect_mask = np.zeros((H, W), dtype=np.float64)
    if remove_mask is None:
        remove_mask = np.zeros((H, W), dtype=np.float64)

    protect_mask = protect_mask.astype(np.float64) * FACTOR
    remove_mask = remove_mask.astype(np.float64) * -FACTOR

    U = np.roll(gray, 1, axis=0)
    R = np.roll(gray,-1, axis=1)
    L = np.roll(gray,1, axis=1)
    CU = np.abs(R - L)
    CL = CU + np.abs(U - L)
    CR = CU + np.abs(U - R)

    M[0] = M[0] + protect_mask[0] + remove_mask[0]

    for i in range(1, H):
        MU = M[i - 1]
        ML = np.roll(MU, 1)
        MR = np.roll(MU, -1)
        MU += CU[i]
        ML += CL[i]
        MR += CR[i]

        MULR = np.array([ML, MU, MR])
        backtrack_path[i] = np.argmin(MULR, axis=0)
        M[i] = protect_mask[i] + remove_mask[i] + np.choose(backtrack_path[i], MULR)
    
    return M, backtrack_path - 1

########################################
# SEAM FUNCTIONS
########################################

def remove_path(img, path, boolean_path=None, remove_mask=None, protection_mask=None):
    """
    Removes a seam path from an image and updates associated masks.
    
    Args:
        img (numpy.ndarray): Input image
        path (list): List of (y,x) coordinates of seam
        boolean_path (numpy.ndarray, optional): Boolean mask of seam
        remove_mask (numpy.ndarray, optional): Mask of pixels to remove
        protection_mask (numpy.ndarray, optional): Mask of protected pixels
        
    Returns:
        tuple: (new image, new remove mask, new protection mask)
    """

    H, W, C = img.shape 
    new_img = np.zeros((H, W - 1, C), dtype=np.float64)
    new_remove_mask = None 
    new_protection_mask = None
    
    if remove_mask is not None:
        new_remove_mask = np.zeros((H, W - 1), dtype=np.float64)
    if protection_mask is not None:
        new_protection_mask = np.zeros((H, W - 1), dtype=np.float64)

    for y, x in path:
        new_img[y, :x, :] = img[y, :x, :]
        new_img[y, x:, :] = img[y, x+1:, :]
        if remove_mask is not None:
            new_remove_mask[y, :x] = remove_mask[y, :x]
            new_remove_mask[y, x:] = remove_mask[y, x+1:]
        if protection_mask is not None:
            new_protection_mask[y, :x] = protection_mask[y, :x]
            new_protection_mask[y, x:] = protection_mask[y, x+1:]

    return new_img, new_remove_mask, new_protection_mask


def insert_seam(img, path, boolean_path=None, protection_mask=None):
    """
    Inserts a new seam path into an image.
    
    Args:
        img (numpy.ndarray): Input image
        path (list): List of (y,x) coordinates where to insert seam
        boolean_path (numpy.ndarray, optional): Boolean mask of seam
        protection_mask (numpy.ndarray, optional): Mask of protected pixels
        
    Returns:
        tuple: (new image with inserted seam, new protection mask)
    """

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

def add_seams(img, num_seams=1, protect_mask=None):
    """
    Adds multiple seams to an image.
    
    Args:
        img (numpy.ndarray): Input image
        num_seams (int): Number of seams to add
        protect_mask (numpy.ndarray, optional): Mask of protected pixels
        
    Returns:
        tuple: (image with added seams, updated protection mask)
    """
    new_img = img.copy()
    new_mask = protect_mask.copy() if protect_mask is not None else None

    # Store seams and their insertion positions
    removed_seams = []
    tmp_img = img.copy()
    tmp_mask = protect_mask.copy() if protect_mask is not None else np.zeros((img.shape[0], img.shape[1]), dtype=np.float64)
    
    # Find and store all seams first
    for i in range(num_seams):
        cum_energyy, backtrack_path = cum_energy(tmp_img, protect_mask=tmp_mask)
        boolean_path, coords_path = getMinPathMask(backtrack_path, cum_energyy)
        tmp_mask += boolean_path.astype(np.float64)
        removed_seams.append(coords_path)

    # Insert the seams
    for seam in removed_seams:
        new_img, new_mask = insert_seam(new_img, seam, protection_mask=new_mask)
        
    return new_img, new_mask

def getMinPathMask(backtrack_path, cum_energy):
    """
    Finds the minimum energy seam path using backtracking.
    
    Args:
        backtrack_path (numpy.ndarray): Matrix of backtracking indices
        cum_energy (numpy.ndarray): Cumulative energy matrix
        
    Returns:
        tuple: (boolean mask of seam path, coordinates of seam path)
    """

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

def remove_seams(img, num_seams=0, protect_mask=None, remove_mask=None):
    """
    Removes multiple seams from an image.
    
    Args:
        img (numpy.ndarray): Input image
        num_seams (int): Number of seams to remove
        protect_mask (numpy.ndarray, optional): Mask of protected pixels
        remove_mask (numpy.ndarray, optional): Mask of pixels to remove
        
    Returns:
        tuple: (image with seams removed, list of removed seam paths)
    """
    if img is None:
        return img, []
        
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
        
    removed_seams = []
    new_img = img.copy()

    curr_protect_mask = protect_mask.copy() if protect_mask is not None else None
    curr_remove_mask = remove_mask.copy() if remove_mask is not None else None
    
    if num_seams == 0: 
        while np.any(curr_remove_mask):
            cum_energyy, backtrack_path = cum_energy(gray, protect_mask=curr_protect_mask, remove_mask=curr_remove_mask)
            boolean_path, coords_path = getMinPathMask(backtrack_path, cum_energyy)
            new_img, curr_remove_mask, curr_protect_mask = remove_path(new_img, 
                                                                    remove_mask=curr_remove_mask,
                                                                    protection_mask=curr_protect_mask, 
                                                                    boolean_path=boolean_path,
                                                                    path=coords_path)
            gray = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            removed_seams.append(coords_path)
    else:
        for i in range(num_seams):
            cum_energyy, backtrack_path = cum_energy(gray, protect_mask=curr_protect_mask, remove_mask=curr_remove_mask)
            boolean_path, coords_path = getMinPathMask(backtrack_path, cum_energyy)
            new_img, curr_remove_mask, curr_protect_mask = remove_path(new_img, 
                                                                    remove_mask=curr_remove_mask,
                                                                    protection_mask=curr_protect_mask, 
                                                                    boolean_path=boolean_path,
                                                                    path=coords_path)
            gray = cv2.cvtColor(new_img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            removed_seams.append(coords_path)

    return new_img, removed_seams

########################################
# MAIN FUNCTIONS
########################################

def remove(mainImg, remove_mask, protect_mask=None):
    """
    Main function to remove an object from an image using seam carving.
    
    Args:
        mainImg (numpy.ndarray): Input image
        remove_mask (numpy.ndarray): Mask indicating object to remove
        
    Returns:
        tuple: (image with object removed, list of removed seam paths)
    """
    w, h = get_masked_obj_dimensions(remove_mask)

    if h < w:
        mainImg = np.rot90(mainImg)
        remove_mask = np.rot90(remove_mask)
    
    removed_img, removed_seams  = remove_seams(img = mainImg, remove_mask=remove_mask, protect_mask=protect_mask)

    if h < w:
        mainImg = np.rot90(mainImg, -1)
        remove_mask = np.rot90(remove_mask, -1)
        removed_img = np.rot90(removed_img, -1)

    return removed_img, removed_seams