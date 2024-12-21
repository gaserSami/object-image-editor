import numpy as np
import cv2
from scipy.signal import convolve2d
import time
import pyopencl as cl

class Inpainter:
    def __init__(self, image, omega, patch_size=9):
        """
        Initialize the inpainting algorithm.
        
        Args:
            image: Input image to be inpainted
            omega: Binary mask where 1 indicates areas to be inpainted
            patch_size: Size of patches used for inpainting (default: 9)
        """
        self.image = image
        self.patch_size = patch_size
        self.omega_3D = omega
        self.half_size = patch_size // 2
        self.tmp_boundary_confidence = None

        # Initialize OpenCL context and compile kernels
        self._init_opencl()

    def _init_opencl(self):
        """Initialize OpenCL context and compile kernels"""
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, """
        // Kernel to compute the sum of squared differences (SSD) between patches
        __kernel void compute_ssd(
            __global const float *image_lab,  // Input image in LAB color space
            __global const float *target_patch,  // Target patch to compare against
            __global const float *target_mask,  // Mask for the target patch
            __global float *ssd,  // Output SSD values
            const int img_height,  // Height of the image
            const int img_width,  // Width of the image
            const int patch_height,  // Height of the patch
            const int patch_width  // Width of the patch
        ) {
            int gid = get_global_id(0);
            int row = gid / (img_width - patch_width + 1);
            int col = gid % (img_width - patch_width + 1);
            
            // Return if the row is out of bounds
            if (row >= (img_height - patch_height + 1)) return;
            
            float sum = 0.0f;
            
            // Iterate over each LAB channel
            for (int c = 0; c < 3; c++) {
                // Iterate over each pixel in the patch
                for (int i = 0; i < patch_height; i++) {
                    for (int j = 0; j < patch_width; j++) {
                        int img_idx = ((row + i) * img_width + (col + j)) * 3 + c;
                        int patch_idx = (i * patch_width + j) * 3 + c;
                        int mask_idx = i * patch_width + j;
                        
                        // Calculate the difference and accumulate the squared difference
                        float diff = (image_lab[img_idx] * target_mask[mask_idx] - 
                                    target_patch[patch_idx]);
                        sum += diff * diff;
                    }
                }
            }
            
            // Store the computed SSD value
            ssd[row * (img_width - patch_width + 1) + col] = sum;
        }

        // Kernel to compute gradients and normals at the boundary
        __kernel void compute_gradients_and_normals(
            __global const float *image,  // Input image
            __global const float *omega,  // Mask indicating inpainting region
            __global float *boundary_gx,  // Output gradient in x direction
            __global float *boundary_gy,  // Output gradient in y direction
            __global float *boundary_nx,  // Output normal in x direction
            __global float *boundary_ny,  // Output normal in y direction
            __global const char *fill_front,  // Fill front mask
            const int height,  // Height of the image
            const int width  // Width of the image
        ) {
            int gid = get_global_id(0);
            int y = gid / width;
            int x = gid % width;
            
            // Return if out of bounds or not on the fill front
            if (y >= height || x >= width || fill_front[y * width + x] == 0) return;
            
            float grad_x = 0, grad_y = 0, nx = 0, ny = 0;
            
            // Compute gradients in x direction
            if (x + 1 < width && omega[(y * width + (x + 1)) * 3] == 0) {
                grad_x = image[(y * width + (x + 1)) * 3] - image[(y * width + x) * 3];
            } else if (x - 1 >= 0 && omega[(y * width + (x - 1)) * 3] == 0) {
                grad_x = image[(y * width + x) * 3] - image[(y * width + (x - 1)) * 3];
            }
            
            // Compute gradients in y direction
            if (y + 1 < height && omega[((y + 1) * width + x) * 3] == 0) {
                grad_y = image[((y + 1) * width + x) * 3] - image[(y * width + x) * 3];
            } else if (y - 1 >= 0 && omega[((y - 1) * width + x) * 3] == 0) {
                grad_y = image[(y * width + x) * 3] - image[((y - 1) * width + x) * 3];
            }
            
            // Compute normals in x direction
            if (x + 1 < width && omega[(y * width + (x + 1)) * 3] == 1) {
                nx = omega[(y * width + (x + 1)) * 3] - omega[(y * width + x) * 3];
            } else if (x - 1 >= 0 && omega[(y * width + (x - 1)) * 3] == 1) {
                nx = omega[(y * width + x) * 3] - omega[(y * width + (x - 1)) * 3];
            }
            
            // Compute normals in y direction
            if (y + 1 < height && omega[((y + 1) * width + x) * 3] == 1) {
                ny = omega[((y + 1) * width + x) * 3] - omega[(y * width + x) * 3];
            } else if (y - 1 >= 0 && omega[((y - 1) * width + x) * 3] == 1) {
                ny = omega[(y * width + x) * 3] - omega[((y - 1) * width + x) * 3];
            }
            
            // Store computed gradients and normals
            boundary_gx[y * width + x] = grad_x;
            boundary_gy[y * width + x] = grad_y;
            boundary_nx[y * width + x] = nx;
            boundary_ny[y * width + x] = ny;
        }

        // Kernel to update confidence values
        __kernel void update_confidence(
            __global const float *confidence,  // Current confidence values
            __global const float *mask,  // Mask indicating inpainting region
            __global float *output,  // Output updated confidence values
            const int height,  // Height of the image
            const int width,  // Width of the image
            const int patch_size,  // Size of the patch
            const int target_row,  // Row of the target patch
            const int target_col  // Column of the target patch
        ) {
            int gid = get_global_id(0);
            int local_row = gid / patch_size;
            int local_col = gid % patch_size;
            
            // Return if local coordinates are out of bounds
            if (local_row >= patch_size || local_col >= patch_size) return;
            
            int half_size = patch_size / 2;
            int global_row = target_row - half_size + local_row;
            int global_col = target_col - half_size + local_col;
            
            // Return if global coordinates are out of bounds
            if (global_row < 0 || global_row >= height || 
                global_col < 0 || global_col >= width) return;
            
            float sum = 0.0f;
            int count = 0;
            
            // Compute average confidence in the patch
            for (int i = -half_size; i <= half_size; i++) {
                for (int j = -half_size; j <= half_size; j++) {
                    int r = global_row + i;
                    int c = global_col + j;
                    
                    if (r >= 0 && r < height && c >= 0 && c < width) {
                        sum += confidence[r * width + c];
                        count++;
                    }
                }
            }
            
            float avg_confidence = sum / (float)count;
            int mask_idx = global_row * width + global_col;
            int out_idx = local_row * patch_size + local_col;
            
            // Update confidence using the mask
            output[out_idx] = confidence[mask_idx] * mask[mask_idx] + 
                             avg_confidence * (1.0f - mask[mask_idx]);
        }
        """).build()

    def initialize(self):
        """Prepare data structures for inpainting"""
        
        # Binarize the omega_3D mask to ensure it's in float64 format
        self.omega_3D = (self.omega_3D > 0).astype(np.float64)
        
        # Create the inverse mask (non-masked areas)
        self.mask_3D = 1 - self.omega_3D
        
        # Calculate the source region, which is the non-masked area in the first channel
        self.source_region = 1 - self.omega_3D[:, :, 0]
        
        # Define a kernel for convolution to find valid patch centers
        kernel = np.ones((self.patch_size, self.patch_size))
        
        # Convolve the source region with the kernel to find valid patch centers
        self.source_centers = convolve2d(
            self.source_region, 
            kernel, 
            mode="same", 
            boundary="fill", 
            fillvalue=0
        ) / (self.patch_size**2)
        
        # Convert source centers to a boolean array where 1 indicates a valid center
        self.source_centers = (self.source_centers == 1)
        
        # Initialize the image by applying the mask and converting to float64
        self.image = self.image.astype(np.float64) * self.mask_3D
        
        # Initialize priority and confidence maps
        self.priority = np.zeros(self.image.shape[:2], dtype=np.float64)
        self.confidence = self.mask_3D[:, :, 0].copy()

    def compute_fill_front(self):
        """
        Compute the fill front (boundary between known and unknown regions)
        using morphological operations.
        """
        # Define a kernel for dilation operation
        kernel = np.array([[0, 1, 0],
                           [1, 1, 1],
                           [0, 1, 0]], dtype=np.uint8)
        
        # Perform dilation on the omega mask to expand the known region
        dilated_omega = cv2.dilate(self.omega_3D[:, :, 0], kernel)
        
        # Compute the fill front by subtracting the original omega from the dilated version
        self.fill_front = dilated_omega - self.omega_3D[:, :, 0]

    def compute_priorites(self):
        # Get boundary pixels coordinates
        boundary_coords = np.where(self.fill_front == 0)
        
        # Compute bounding box around boundary pixels with padding
        min_y, max_y = np.min(boundary_coords[0]) - self.patch_size, np.max(boundary_coords[0]) + self.patch_size
        min_x, max_x = np.min(boundary_coords[1]) - self.patch_size, np.max(boundary_coords[1]) + self.patch_size
        
        # Clamp to image boundaries
        min_y, max_y = max(0, min_y), min(self.image.shape[0], max_y)
        min_x, max_x = max(0, min_x), min(self.image.shape[1], max_x)
        
        # Extract region of interest
        roi_height = max_y - min_y
        roi_width = max_x - min_x
        
        # Compute confidence term only for ROI
        kernel = np.ones((self.patch_size, self.patch_size))
        
        roi_confidence = self.confidence[min_y:max_y, min_x:max_x]
        roi_fill_front = self.fill_front[min_y:max_y, min_x:max_x]
        
        boundary_confidence = convolve2d(roi_confidence, kernel, mode="same") / (self.patch_size**2)
        boundary_confidence = np.where(roi_fill_front, boundary_confidence, 0)
        self.tmp_boundary_confidence = boundary_confidence
        
        # Initialize ROI output arrays
        boundary_gx = np.zeros((roi_height, roi_width), dtype=np.float32)
        boundary_gy = np.zeros((roi_height, roi_width), dtype=np.float32)
        boundary_nx = np.zeros((roi_height, roi_width), dtype=np.float32)
        boundary_ny = np.zeros((roi_height, roi_width), dtype=np.float32)
        
        # Create OpenCL buffers for ROI
        mf = cl.mem_flags
        image_roi = self.image[min_y:max_y, min_x:max_x]
        omega_roi = self.omega_3D[min_y:max_y, min_x:max_x]
        
        image_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=image_roi.astype(np.float32))
        omega_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=omega_roi.astype(np.float32))
        fill_front_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=roi_fill_front.astype(np.int8))
        gx_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, boundary_gx.nbytes)
        gy_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, boundary_gy.nbytes)
        nx_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, boundary_nx.nbytes)
        ny_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, boundary_ny.nbytes)
        
        # Execute kernel on ROI
        global_size = (roi_height * roi_width,)
        self.prg.compute_gradients_and_normals(
            self.queue, global_size, None,
            image_buf, omega_buf, gx_buf, gy_buf, nx_buf, ny_buf, fill_front_buf,
            np.int32(roi_height), np.int32(roi_width)
        )
        
        # Get results
        cl.enqueue_copy(self.queue, boundary_gx, gx_buf)
        cl.enqueue_copy(self.queue, boundary_gy, gy_buf)
        cl.enqueue_copy(self.queue, boundary_nx, nx_buf)
        cl.enqueue_copy(self.queue, boundary_ny, ny_buf)
        
        # Rest of the computations using numpy
        magnitude = np.sqrt(boundary_gx**2 + boundary_gy**2)
        magnitude = np.where(magnitude == 0, 1, magnitude)
        
        isophote_x = -boundary_gy / magnitude
        isophote_y = boundary_gx / magnitude
        
        dot_product = (isophote_x * boundary_nx + isophote_y * boundary_ny)
        data_term = np.abs(dot_product) / 255.0
        
        # Initialize full priority matrix
        self.priority.fill(-1)
        
        # Update only the ROI in the priority matrix
        roi_priority = data_term * boundary_confidence
        roi_priority = np.where(roi_fill_front, roi_priority, -1)
        self.priority[min_y:max_y, min_x:max_x] = roi_priority

    def find_max_prio_patch(self):
        """
        Find the coordinates of the patch with the maximum priority.
        
        Returns:
            max_coords: Tuple of (row, column) coordinates of the patch with the highest priority.
        """
        # Find the index of the maximum priority value
        max_index = np.argmax(self.priority)
        
        # Convert the flat index to 2D coordinates
        max_coords = np.unravel_index(max_index, self.priority.shape)
        
        return max_coords

    def find_exemplar(self, target_patch, target_mask_patch, target_coords):
        """
        Find the exemplar patch with the minimum SSD (Sum of Squared Differences).
        
        Args:
            target_patch: The patch to be inpainted.
            target_mask_patch: The mask for the target patch.
            target_coords: Coordinates of the target patch.
        
        Returns:
            min_coords: Tuple of (row, column) coordinates of the exemplar patch.
        """
        # Compute the SSD between the target patch and all possible source patches
        ssd = self.SSD(target_patch, target_mask_patch, target_coords)
        
        # Set SSD to infinity for invalid source centers
        ssd[self.source_centers == 0] = np.inf
        
        # Find the coordinates of the minimum SSD value
        min_coords = np.unravel_index(np.argmin(ssd), ssd.shape)
        
        return min_coords

    def SSD(self, target_patch, target_mask, target_coords):
        """
        OpenCL-accelerated SSD calculation using BGR color space
        """
        # Use BGR values directly without color space conversion
        target_patch = target_patch.astype(np.float32) / 255.0
        target_patch = target_patch * target_mask

        # Get shapes
        patch_height, patch_width = target_patch.shape[:2]
        img_height, img_width = self.image.shape[:2]
        
        # Calculate output dimensions
        out_height = img_height - patch_height + 1
        out_width = img_width - patch_width + 1
        
        # Create output array
        ssd = np.zeros((out_height, out_width), dtype=np.float32)

        # Create OpenCL buffers
        mf = cl.mem_flags
        image_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                             hostbuf=(self.image.astype(np.float32) / 255.0))
        target_patch_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                    hostbuf=target_patch.astype(np.float32))
        target_mask_buf = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, 
                                   hostbuf=target_mask[:,:,0].astype(np.float32))
        ssd_buf = cl.Buffer(self.ctx, mf.WRITE_ONLY, ssd.nbytes)

        # Execute kernel
        global_size = (out_height * out_width,)
        local_size = None  # Let OpenCL choose the work-group size

        self.prg.compute_ssd(self.queue, global_size, local_size,
                           image_buf, target_patch_buf, target_mask_buf, ssd_buf,
                           np.int32(img_height), np.int32(img_width),
                           np.int32(patch_height), np.int32(patch_width))

        # Get results
        cl.enqueue_copy(self.queue, ssd, ssd_buf)

        # Pad the result
        pad_height = patch_height - 1
        pad_width = patch_width - 1
        padded_ssd = np.pad(
            ssd,
            ((pad_height//2, (pad_height+1)//2),
             (pad_width//2, (pad_width+1)//2)),
            mode='constant',
            constant_values=np.inf
        )

        return padded_ssd

    def copy_exemplar_data(self, target_patch, target_mask_patch, target_coords, source_coords):
        """
        Copy data from the exemplar patch to the target patch location.
        
        Args:
            target_patch: The patch to be inpainted.
            target_mask_patch: The mask for the target patch.
            target_coords: Coordinates of the target patch.
            source_coords: Coordinates of the exemplar patch.
        """
        # Calculate half the patch size for indexing
        half_patch_size = self.patch_size // 2
        
        # Extract the source patch from the image using source coordinates
        source_row, source_col = source_coords
        source_patch = self.image[
            -half_patch_size + source_row:half_patch_size + source_row + 1, 
            -half_patch_size + source_col:half_patch_size + source_col + 1
        ]
        
        # Create a new patch by combining the target and source patches using the mask
        new_patch = target_patch * target_mask_patch + source_patch * (1 - target_mask_patch)
        
        # Place the new patch into the target location in the image
        target_row, target_col = target_coords
        self.image[
            -half_patch_size + target_row:half_patch_size + target_row + 1, 
            -half_patch_size + target_col:half_patch_size + target_col + 1
        ] = new_patch
        
        # Update the omega mask to indicate the region has been filled
        self.omega_3D[
            -half_patch_size + target_row:half_patch_size + target_row + 1, 
            -half_patch_size + target_col:half_patch_size + target_col + 1
        ] = 0
        
        # Update the inverse mask
        self.mask_3D = 1 - self.omega_3D

    def update_confidence(self, target_coords, target_mask_patch):
        """
        Update the confidence values for the target patch area.
        
        Args:
            target_coords: Coordinates of the target patch.
            target_mask_patch: The mask for the target patch.
        """
        # Extract target row and column from coordinates
        target_row, target_col = target_coords
        
        # Retrieve the new confidence value from the temporary boundary confidence map
        new_confidence = self.tmp_boundary_confidence[target_row, target_col]
        
        # Extract the current confidence patch from the confidence map
        new_confidence_patch = self.confidence[
            target_row - self.half_size:target_row + self.half_size + 1, 
            target_col - self.half_size:target_col + self.half_size + 1
        ]
        
        # Update the confidence patch using the target mask
        new_confidence_patch = (
            new_confidence_patch * target_mask_patch[:, :, 0] + 
            new_confidence * (1 - target_mask_patch[:, :, 0])
        )
        
        # Place the updated confidence patch back into the confidence map
        self.confidence[
            target_row - self.half_size:target_row + self.half_size + 1, 
            target_col - self.half_size:target_col + self.half_size + 1
        ] = new_confidence_patch

    def inpaint(self):
        """
        Perform the inpainting process on the image.
        """
        # Continue inpainting while there are regions to fill
        while np.any(self.omega_3D):
            # Compute the fill front (boundary of the inpainting region)
            self.compute_fill_front()

            # Compute priorities for the fill front
            self.compute_priorites()

            # Find the patch with the maximum priority
            target_coords = self.find_max_prio_patch()

            # Extract the target patch and its mask
            target_row, target_col = target_coords
            target_patch = self.image[
                -self.half_size + target_row:self.half_size + target_row + 1, 
                -self.half_size + target_col:self.half_size + target_col + 1
            ]
            target_mask_patch = self.mask_3D[
                -self.half_size + target_row:self.half_size + target_row + 1, 
                -self.half_size + target_col:self.half_size + target_col + 1
            ]

            # Find the exemplar patch with the minimum SSD
            source_coords = self.find_exemplar(
                target_patch=target_patch, 
                target_mask_patch=target_mask_patch, 
                target_coords=target_coords
            )

            # Copy data from the exemplar patch to the target patch
            self.copy_exemplar_data(
                target_patch=target_patch, 
                target_mask_patch=target_mask_patch, 
                target_coords=target_coords, 
                source_coords=source_coords
            )

            # Update the confidence map for the target patch
            self.update_confidence(
                target_coords=target_coords, 
                target_mask_patch=target_mask_patch
            )
