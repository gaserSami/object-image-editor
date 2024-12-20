import cv2
import numpy as np

def remove_seams(img, num_seams=0, protect_mask=None, remove_mask=None, forward=True):
    """Main function to remove seams either by count or based on removal mask.

    Parameters:
        img: Input image from which seams are to be removed.
        num_seams: Number of seams to remove. If 0, remove based on the removal mask.
        protect_mask: Mask indicating areas to protect from seam removal.
        remove_mask: Mask indicating areas to prioritize for seam removal.
        forward: Boolean indicating whether to use forward energy computation.

    Returns:
        new_img: Image after seam removal.
        removed_seams: List of removed seam paths.
    """
    # Return early if the image is None
    if img is None:
        return img, []

    # Initialize variables
    new_img = img.copy()
    curr_protect_mask = protect_mask.copy() if protect_mask is not None else None
    curr_remove_mask = remove_mask.copy() if remove_mask is not None else None

    # Remove seams based on the number of seams or removal mask
    if num_seams == 0:
        # Remove seams until the removal mask is empty
        while np.any(curr_remove_mask):
            new_img, curr_remove_mask, curr_protect_mask, _ = _remove_single_seam(
                img=new_img, protect_mask=curr_protect_mask, remove_mask=curr_remove_mask, forward=forward)
    else:
        # Remove a specified number of seams
        for _ in range(num_seams):
            new_img, curr_remove_mask, curr_protect_mask, _ = _remove_single_seam(
                img=new_img, protect_mask=curr_protect_mask, remove_mask=curr_remove_mask, forward=forward)

    return new_img

def _remove_single_seam(img, protect_mask, remove_mask, forward):
    """Helper function to remove a single seam.

    Parameters:
        img: The current image from which a seam is to be removed.
        protect_mask: Mask indicating areas to protect from seam removal.
        remove_mask: Mask indicating areas to prioritize for seam removal.
        forward: Boolean indicating whether to use forward energy computation.

    Returns:
        new_img: Image after a single seam removal.
        new_remove_mask: Updated remove mask after seam removal.
        new_protect_mask: Updated protect mask after seam removal.
        coords_path: Coordinates of the removed seam path.
    """
    # Compute cumulative energy map and backtrack map
    cum_energy_map, backtrack_map = _cum_energy(
        img, protect_mask=protect_mask, remove_mask=remove_mask, forward=forward
    )

    # Determine the path of the minimum energy seam
    coords_path = _get_min_path(backtrack_map, cum_energy_map)

    # Remove the identified seam path from the image and masks
    new_imgs = _remove_path_from_imgs(
        imgs=[img, remove_mask, protect_mask], path=coords_path
    )

    # Return the updated image, masks, and seam path
    return new_imgs[0], new_imgs[1], new_imgs[2], coords_path

def add_seams(img, num_seams=1, protect_mask=None, forward=True):
    """Main function to add seams to enlarge an image."""
    orig_img = img.copy().astype(np.float64)
    H, W = orig_img.shape[:2]
    C = orig_img.shape[2] if len(orig_img.shape) == 3 else 1

    # Prepare a mask for removed seams
    removed_seams_mask = np.zeros((H, W), dtype=np.float64)

    # Temporary copies for seam removal
    tmp_img = img.copy()
    tmp_mask = protect_mask.copy() if protect_mask is not None else np.zeros((H, W), dtype=np.float64)
    tmp_idx_map = np.tile(np.arange(W), (H, 1))

    # Array to store the removed seams. Each seam is (H,2): (row, original_col)
    removed_seams = np.empty((num_seams, H, 2), dtype=np.int64)

    # Identify all seams to be inserted
    for i in range(num_seams):
        cum_energy_map, backtrack_map = _cum_energy(tmp_img, protect_mask=tmp_mask, forward=forward)
        coords_path = _get_min_path(backtrack_map, cum_energy_map)

        # Map coords_path to original indices BEFORE removing the path
        updated_path = coords_path.copy()
        updated_path[:, 1] = tmp_idx_map[coords_path[:, 0], coords_path[:, 1]]

        # Now remove the path from tmp_img and update tmp_idx_map
        new_imgs = _remove_path_from_imgs(
            imgs=[tmp_img, tmp_mask, tmp_idx_map],
            path=coords_path,
        )

        # Update references
        tmp_img = new_imgs[0]
        tmp_mask = new_imgs[1]
        tmp_idx_map = new_imgs[2]

        # Update other references
        removed_seams[i] = updated_path
        removed_seams_mask[updated_path[:, 0], updated_path[:, 1]] = 1

    # Build the final image by inserting seams
    final_width = W + num_seams
    final_img = np.zeros((H, final_width, C), dtype=np.float64)

    # For each row, determine the columns where seams were removed
    # and then re-insert them by duplicating pixels
    for h in range(H):
        # Collect all seam column indices for this row
        # These are the original columns where seams were removed
        seam_cols = []
        for i in range(num_seams):
            seam_cols.append(removed_seams[i, h, 1])
        seam_cols = sorted(seam_cols)

        # We'll rebuild this row
        row_buffer = np.zeros((final_width, C), dtype=np.float64)
        
        # Indices to track insertion
        original_col_idx = 0
        final_col_idx = 0
        seam_idx = 0  # which seam index we are currently inserting

        while original_col_idx < W:
            # Copy original pixel to final image
            row_buffer[final_col_idx, :] = orig_img[h, original_col_idx, :]

            # Check if the next seam to insert is at this column
            if seam_idx < len(seam_cols) and seam_cols[seam_idx] == original_col_idx:
                # We need to insert a pixel right after this column
                # Average between current pixel and the next one (or handle boundary)
                if original_col_idx < W - 1:
                    new_val = (orig_img[h, original_col_idx, :] + orig_img[h, original_col_idx + 1, :]) / 2.0
                else:
                    # At the right boundary, average with the previous pixel
                    # This case usually doesn't happen with seam insertion, but just in case:
                    new_val = (orig_img[h, original_col_idx, :] + orig_img[h, original_col_idx - 1, :]) / 2.0

                final_col_idx += 1
                row_buffer[final_col_idx, :] = new_val
                seam_idx += 1

            original_col_idx += 1
            final_col_idx += 1

        final_img[h, :, :] = row_buffer

    return final_img.astype(np.uint8)

def _compute_grad(img):
    """Computes gradient energy across all color channels and sums them.

    Parameters:
        img: Input image, can be grayscale or color.

    Returns:
        energy: Normalized gradient energy map of the image.
    """
    # Check if the image is grayscale
    if len(img.shape) == 2:
        # Compute gradient energy for a grayscale image
        grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        return np.abs(grad_x) + np.abs(grad_y)

    # Initialize energy map for a color image
    energy = np.zeros(img.shape[:2], dtype=np.float64)

    # Compute gradient energy for each color channel and sum them
    for channel in range(img.shape[2]):
        grad_x = cv2.Sobel(img[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(img[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
        energy += np.abs(grad_x) + np.abs(grad_y)

    # Normalize the energy map to the range [0, 1]
    min_energy = np.min(energy)
    max_energy = np.max(energy)
    energy = (energy - min_energy) / (max_energy - min_energy)

    return energy

def _cum_energy(img, protect_mask=None, remove_mask=None, forward=True):
    """Computes cumulative energy matrix"""
    if not forward:
        return _backward_energy(img, protect_mask, remove_mask)
    return _forward_energy(img, protect_mask, remove_mask)

def _backward_energy(img, protect_mask, remove_mask):
    """Helper function for backward energy computation.

    Parameters:
        img: Input image for which the energy map is computed.
        protect_mask: Mask indicating areas to protect from seam removal.
        remove_mask: Mask indicating areas to prioritize for seam removal.

    Returns:
        cum_energy_map: Cumulative energy map.
        backtrack_map: Map used to backtrack the minimum energy seam.
    """
    # Compute the gradient energy of the image
    energy = _compute_grad(img)
    
    # Initialize dimensions and cumulative energy map
    H, W = energy.shape
    cum_energy_map = energy.copy()
    backtrack_map = np.empty((H, W), dtype=np.int32)

    # Get the factor
    factor = _get_factor()

    # Apply protect mask to the energy map
    if protect_mask is not None:
        protect_mask = np.where(protect_mask > 0, factor, protect_mask)
        cum_energy_map += protect_mask
    # Apply remove mask to the energy map
    if remove_mask is not None:
        remove_mask = np.where(remove_mask > 0, -factor * 1e2, remove_mask)
        cum_energy_map += remove_mask
    
    # Compute cumulative energy and backtrack map
    for i in range(1, H):
        # Pad the previous row to handle edge cases
        prev_row = np.pad(cum_energy_map[i - 1], (1, 1), mode='constant', constant_values=np.inf)
        left = prev_row[:-2]
        center = prev_row[1:-1]
        right = prev_row[2:]
        
        # Stack left, center, and right for comparison
        lcr = np.array([left, center, right])
        
        # Determine the minimum energy path
        min_choices = np.argmin(lcr, axis=0)  # 0 = left, 1 = center, 2 = right
        backtrack_map[i] = min_choices - 1
        
        # Update the cumulative energy map
        cum_energy_map[i] += np.choose(min_choices, lcr)

    return cum_energy_map, backtrack_map

def _forward_energy(img, protect_mask, remove_mask):
    """Helper function for forward energy computation.

    Parameters:
        img: Input image for which the energy map is computed.
        protect_mask: Mask indicating areas to protect from seam removal.
        remove_mask: Mask indicating areas to prioritize for seam removal.

    Returns:
        cum_energy_map: Cumulative energy map.
        backtrack_map: Map used to backtrack the minimum energy seam.
    """
    # Convert image to grayscale if it's a color image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

    # Pad the grayscale image to handle edge cases
    gray_paded = np.pad(gray, pad_width=1, mode='edge')
    H, W = gray.shape

    # Initialize backtrack map and cumulative energy map
    backtrack_map = np.empty((H, W), dtype=np.int32)
    cum_energy_map = np.zeros((H, W), dtype=np.float64)

    # Compute initial energy for the first row
    curr_r = gray_paded[0, 2:]
    curr_l = gray_paded[0, :-2]
    CU = np.abs(curr_r - curr_l)
    cum_energy_map[0] = CU

    # Get the factor
    factor = _get_factor()

    # Apply protect mask to the energy map  
    if protect_mask is not None:
        protect_mask = np.where(protect_mask > 0, factor, protect_mask)
        cum_energy_map += protect_mask
    # Apply remove mask to the energy map
    if remove_mask is not None:
        remove_mask = np.where(remove_mask > 0, -factor * 1e2, remove_mask)
        cum_energy_map += remove_mask

    # Compute cumulative energy and backtrack map for each row
    for i in range(1, H):
        curr_r = gray_paded[i, 2:]
        curr_l = gray_paded[i, :-2]
        CU = np.abs(curr_r - curr_l)
        prev_row_U = gray[i - 1]
        CL = np.abs(prev_row_U - curr_l) + CU
        CR = np.abs(prev_row_U - curr_r) + CU

        # Pad the previous row of the cumulative energy map
        prev_row = np.pad(cum_energy_map[i - 1], (1, 1), mode='constant', constant_values=np.inf)
        left = prev_row[:-2]
        center = prev_row[1:-1]
        right = prev_row[2:]

        # Stack left, center, and right for comparison
        lcr = np.array([left, center, right])
        lcr += np.array([CL, CU, CR])

        # Determine the minimum energy path
        min_choices = np.argmin(lcr, axis=0)  # 0 = left, 1 = center, 2 = right
        backtrack_map[i] = min_choices - 1

        # Update the cumulative energy map
        cum_energy_map[i] += np.choose(min_choices, lcr)

    return cum_energy_map, backtrack_map


def _remove_path_from_imgs(imgs, path):
    """Removes a seam path from image and associated masks.

    Parameters:
        imgs: List of images and masks from which the seam path is to be removed.
        path: Coordinates of the seam path to be removed.

    Returns:
        new_imgs: List of images and masks after seam removal.
    """
    # Initialize new images with one less column for each image
    new_imgs = []
    for img in imgs:
        if img is not None:
            H, W = img.shape[:2]
            if len(img.shape) == 3:
                new_img = np.zeros((H, W - 1, img.shape[2]), dtype=img.dtype)
            else:
                new_img = np.zeros((H, W - 1), dtype=img.dtype)
            new_imgs.append(new_img)
        else:
            new_imgs.append(None)

    # Iterate over each coordinate in the path
    for y, x in path:
        for i, img in enumerate(imgs):
            if img is not None:
                # Handle multi-channel and single-channel images
                if len(img.shape) == 3:
                    new_imgs[i][y, :x, :] = img[y, :x, :]
                    new_imgs[i][y, x:, :] = img[y, x+1:, :]
                else:
                    new_imgs[i][y, :x] = img[y, :x]
                    new_imgs[i][y, x:] = img[y, x+1:]

    return new_imgs

def _get_min_path(backtrack_map, cum_energy_map):
    """Finds the minimum energy seam path.

    Parameters:
        backtrack_map: Map used to backtrack the minimum energy seam.
        cum_energy_map: Cumulative energy map.

    Returns:
        path: Coordinates of the minimum energy seam path.
    """
    # Get the dimensions of the backtrack map
    H, W = backtrack_map.shape

    # Initialize the path array to store seam coordinates
    path = np.empty((H, 2), dtype=np.int64)

    # Start from the position with the minimum energy in the last row
    x = cum_energy_map[-1].argmin()

    # Backtrack from the bottom to the top of the image
    for i in range(H - 1, -1, -1):
        # Store the current position in the path
        path[i] = (i, x)

        # Update x to the next position in the seam path
        x += backtrack_map[i, x]

        # Ensure x stays within image bounds
        x = max(0, min(W - 1, x))

    return path

def _get_masked_obj_dimensions(mask):
    """Gets dimensions of the masked object.

    Parameters:
        mask: A binary mask where the object is marked.

    Returns:
        width: Width of the masked object.
        height: Height of the masked object.
    """
    # Return zero dimensions if the mask is None
    if mask is None:
        return 0, 0

    # Convert the mask to a numpy array
    mask = np.asarray(mask)

    # Find the indices where the mask is non-zero
    y_indices, x_indices = np.where(mask)

    # Check if there are no non-zero indices
    if len(x_indices) == 0 or len(y_indices) == 0:
        return 0, 0

    # Calculate the width and height of the masked object
    width = x_indices.max() - x_indices.min() + 1
    height = y_indices.max() - y_indices.min() + 1

    return width, height


def _compute_rotation_decision(img, remove_mask):
    """Determine if rotation is needed based on gradient energy analysis.

    Parameters:
        img: Input image for analysis.
        remove_mask: Mask indicating the area to be removed.

    Returns:
        bool: True if rotation is needed, False otherwise.
    """
    # Calculate gradients in both horizontal and vertical directions
    grad_x = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate average energy in both directions within the mask
    mask_3d = np.stack([remove_mask] * 3, axis=-1)
    horizontal_energy = np.mean(np.abs(grad_x)[mask_3d == 1])
    vertical_energy = np.mean(np.abs(grad_y)[mask_3d == 1])

    # Get dimensions of the object within the mask
    w, h = _get_masked_obj_dimensions(remove_mask)

    # Consider both energy and dimensions in the decision
    energy_ratio = horizontal_energy / vertical_energy
    dimension_ratio = w / h

    # Rotate if vertical energy is significantly lower and width is larger
    # or if horizontal energy is much stronger despite dimensions
    return energy_ratio > 1.2 or (dimension_ratio > 1.5 and energy_ratio > 0.8)

def remove_object(mainImg, remove_mask, protect_mask=None, forward=True, direction="auto"):
    """
    Main public interface for object removal.

    Parameters:
        mainImg: Input image from which objects are to be removed.
        remove_mask: Mask indicating the area to be removed.
        protect_mask: Mask indicating the area to be protected (optional).
        forward: Boolean indicating whether to use forward energy computation (default: True).
        direction: Direction for seam removal ('vertical', 'horizontal', or 'auto' for automatic detection).

    Returns:
        removed_img: Image after object removal.
        removed_seams: List of removed seam paths.
    """
    # Determine if rotation is needed based on the specified or automatic direction
    if direction != "auto":
        # Use specified direction
        should_rotate = direction.lower() == 'horizontal'
    else:
        # Use automatic direction detection
        should_rotate = _compute_rotation_decision(mainImg, remove_mask)

    # Rotate the image and masks if needed
    if should_rotate:
        mainImg = np.rot90(mainImg)
        remove_mask = np.rot90(remove_mask)
        if protect_mask is not None:
            protect_mask = np.rot90(protect_mask)

    # Perform seam removal
    removed_img = remove_seams(
        img=mainImg, remove_mask=remove_mask, 
        protect_mask=protect_mask, forward=forward
    )

    # Rotate the image back to its original orientation if it was rotated
    if should_rotate:
        removed_img = np.rot90(removed_img, -1)

    return removed_img

def _get_factor():
    return 1e4