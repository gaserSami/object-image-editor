import numpy as np
import cv2
import matplotlib.pyplot as plt
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
        
        # Performance tracking
        self.timing_stats = {
            'compute_fill_front': [],
            'compute_priorities': [],
            'find_max_prio_patch': [],
            'find_exemplar': [],
            'copy_exemplar_data': [],
            'update_confidence': []
        }

        # Initialize OpenCL context and compile kernels
        self._init_opencl()

    def _init_opencl(self):
        """Initialize OpenCL context and compile kernels"""
        self.ctx = cl.create_some_context()
        self.queue = cl.CommandQueue(self.ctx)
        self.prg = cl.Program(self.ctx, """
        __kernel void compute_ssd(
            __global const float *image_lab,
            __global const float *target_patch,
            __global const float *target_mask,
            __global float *ssd,
            const int img_height,
            const int img_width,
            const int patch_height,
            const int patch_width
        ) {
            int gid = get_global_id(0);
            int row = gid / (img_width - patch_width + 1);
            int col = gid % (img_width - patch_width + 1);
            
            if (row >= (img_height - patch_height + 1)) return;
            
            float sum = 0.0f;
            
            // For each LAB channel
            for (int c = 0; c < 3; c++) {
                // For each pixel in patch
                for (int i = 0; i < patch_height; i++) {
                    for (int j = 0; j < patch_width; j++) {
                        int img_idx = ((row + i) * img_width + (col + j)) * 3 + c;
                        int patch_idx = (i * patch_width + j) * 3 + c;
                        int mask_idx = i * patch_width + j;
                        
                        float diff = (image_lab[img_idx] * target_mask[mask_idx] - 
                                    target_patch[patch_idx]);
                        sum += diff * diff;
                    }
                }
            }
            
            ssd[row * (img_width - patch_width + 1) + col] = sum;
        }

        __kernel void compute_gradients_and_normals(
            __global const float *image,
            __global const float *omega,
            __global float *boundary_gx,
            __global float *boundary_gy,
            __global float *boundary_nx,
            __global float *boundary_ny,
            __global const char *fill_front,
            const int height,
            const int width
        ) {
            int gid = get_global_id(0);
            int y = gid / width;
            int x = gid % width;
            
            if (y >= height || x >= width || fill_front[y * width + x] == 0) return;
            
            float grad_x = 0, grad_y = 0, nx = 0, ny = 0;
            
            // Compute gradients
            if (x + 1 < width && omega[(y * width + (x + 1)) * 3] == 0) {
                grad_x = image[(y * width + (x + 1)) * 3] - image[(y * width + x) * 3];
            } else if (x - 1 >= 0 && omega[(y * width + (x - 1)) * 3] == 0) {
                grad_x = image[(y * width + x) * 3] - image[(y * width + (x - 1)) * 3];
            }
            
            if (y + 1 < height && omega[((y + 1) * width + x) * 3] == 0) {
                grad_y = image[((y + 1) * width + x) * 3] - image[(y * width + x) * 3];
            } else if (y - 1 >= 0 && omega[((y - 1) * width + x) * 3] == 0) {
                grad_y = image[(y * width + x) * 3] - image[((y - 1) * width + x) * 3];
            }
            
            // Compute normals
            if (x + 1 < width && omega[(y * width + (x + 1)) * 3] == 1) {
                nx = omega[(y * width + (x + 1)) * 3] - omega[(y * width + x) * 3];
            } else if (x - 1 >= 0 && omega[(y * width + (x - 1)) * 3] == 1) {
                nx = omega[(y * width + x) * 3] - omega[(y * width + (x - 1)) * 3];
            }
            
            if (y + 1 < height && omega[((y + 1) * width + x) * 3] == 1) {
                ny = omega[((y + 1) * width + x) * 3] - omega[(y * width + x) * 3];
            } else if (y - 1 >= 0 && omega[((y - 1) * width + x) * 3] == 1) {
                ny = omega[(y * width + x) * 3] - omega[((y - 1) * width + x) * 3];
            }
            
            boundary_gx[y * width + x] = grad_x;
            boundary_gy[y * width + x] = grad_y;
            boundary_nx[y * width + x] = nx;
            boundary_ny[y * width + x] = ny;
        }

        __kernel void update_confidence(
            __global const float *confidence,
            __global const float *mask,
            __global float *output,
            const int height,
            const int width,
            const int patch_size,
            const int target_row,
            const int target_col
        ) {
            int gid = get_global_id(0);
            int local_row = gid / patch_size;
            int local_col = gid % patch_size;
            
            if (local_row >= patch_size || local_col >= patch_size) return;
            
            int half_size = patch_size / 2;
            int global_row = target_row - half_size + local_row;
            int global_col = target_col - half_size + local_col;
            
            if (global_row < 0 || global_row >= height || 
                global_col < 0 || global_col >= width) return;
            
            float sum = 0.0f;
            int count = 0;
            
            // Compute average confidence in patch
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
            
            // Update confidence using mask
            output[out_idx] = confidence[mask_idx] * mask[mask_idx] + 
                             avg_confidence * (1.0f - mask[mask_idx]);
        }
        """).build()

    def initialize(self):
        """Prepare data structures for inpainting"""
        # Normalize and convert data types
        self.omega_3D = self.omega_3D.astype(np.float64) / 255
        self.mask_3D = 1 - self.omega_3D
        
        # Calculate source region (non-masked areas)
        self.source_region = 1 - self.omega_3D[:, :, 0]
        
        # Find valid patch centers in source region
        kernel = np.ones((self.patch_size, self.patch_size))
        self.source_centers = convolve2d(
            self.source_region, 
            kernel, 
            mode="same", 
            boundary="fill", 
            fillvalue=0
        ) / (self.patch_size**2)
        self.source_centers = (self.source_centers == 1)
        
        # Initialize image and confidence maps
        self.image = self.image.astype(np.float64) * self.mask_3D
        self.priority = np.zeros(self.image.shape[:2], dtype=np.float64)
        self.confidence = self.mask_3D[:,:,0].copy()

    def compute_fill_front(self):
        """
        Compute the fill front (boundary between known and unknown regions)
        using morphological operations
        """
        kernel = np.array([[0, 1, 0],
                          [1, 1, 1],
                          [0, 1, 0]], dtype=np.uint8)
        self.fill_front = cv2.dilate(self.omega_3D[:,:,0], kernel) - self.omega_3D[:,:,0]

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
        center = self.patch_size // 2
        indices = np.arange(self.patch_size)
        dist_matrix = np.abs(indices[:, None] - center) + np.abs(indices - center)
        dist_matrix = dist_matrix / 6
        kernel = dist_matrix
        
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
        self.priority.fill(0)
        
        # Update only the ROI in the priority matrix
        roi_priority = data_term * boundary_confidence
        roi_priority = np.where(roi_fill_front, roi_priority, 0)
        self.priority[min_y:max_y, min_x:max_x] = roi_priority

    def find_max_prio_patch(self):
        max_coords = np.unravel_index(np.argmax(self.priority), self.priority.shape)
        return max_coords

    def find_exemplar(self, target_patch, target_mask_patch, target_coords):
        ssd = self.SSD(target_patch, target_mask_patch, target_coords)
        ssd[self.source_centers == 0] = np.inf
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
        # paste only in the omega region in the mask_patch
        half_patch_size = self.patch_size // 2
        source_row, source_col = source_coords
        source_patch = self.image[-half_patch_size + source_row:half_patch_size + source_row + 1, -half_patch_size + source_col:half_patch_size + source_col + 1]
        new_patch = target_patch * target_mask_patch + source_patch * (1 - target_mask_patch)
        target_row, target_col = target_coords
        self.image[-half_patch_size + target_row:half_patch_size + target_row + 1, -half_patch_size + target_col:half_patch_size + target_col + 1] = new_patch
        self.omega_3D[-half_patch_size + target_row:half_patch_size + target_row + 1, -half_patch_size + target_col:half_patch_size + target_col + 1] = 0
        self.mask_3D = 1 - self.omega_3D

    def update_confidence(self, target_coords, target_mask_patch):
        target_row, target_col = target_coords
        new_confidence = self.tmp_boundary_confidence[target_row, target_col]
        new_confidence_patch = self.confidence[target_row - self.half_size:target_row + self.half_size + 1, target_col - self.half_size:target_col + self.half_size + 1]
        new_confidence_patch = new_confidence_patch * target_mask_patch[:,:,0] + new_confidence * (1 - target_mask_patch[:,:,0])
        self.confidence[target_row - self.half_size:target_row + self.half_size + 1, target_col - self.half_size:target_col + self.half_size + 1] = new_confidence_patch

    def inpaint(self):
        to_save_after_each = 25
        i = 0
        start_time = time.time()
        while np.any(self.omega_3D):
            # DEBUGGING: TO REMOVE
            print(f"omega_3D is any? : {np.any(self.omega_3D)}")
            print(f"sum of omega_3D: {np.sum(self.omega_3D)}")
            t0 = time.time()
            self.compute_fill_front()
            self.timing_stats['compute_fill_front'].append(time.time() - t0)

            t0 = time.time()
            self.compute_priorites()
            self.timing_stats['compute_priorities'].append(time.time() - t0)

            t0 = time.time()
            target_coords = self.find_max_prio_patch()
            self.timing_stats['find_max_prio_patch'].append(time.time() - t0)

            target_row, target_col = target_coords
            target_patch = self.image[-self.half_size + target_row:self.half_size + target_row + 1, -self.half_size + target_col:self.half_size + target_col + 1]
            target_mask_patch = self.mask_3D[-self.half_size + target_row:self.half_size + target_row + 1, -self.half_size + target_col:self.half_size + target_col + 1]

            t0 = time.time()
            source_coords = self.find_exemplar(target_patch=target_patch, target_mask_patch=target_mask_patch, target_coords=target_coords)
            self.timing_stats['find_exemplar'].append(time.time() - t0)

            t0 = time.time()
            self.copy_exemplar_data(target_patch=target_patch, target_mask_patch=target_mask_patch, target_coords=target_coords, source_coords=source_coords)
            self.timing_stats['copy_exemplar_data'].append(time.time() - t0)

            t0 = time.time()
            self.update_confidence(target_coords=target_coords, target_mask_patch=target_mask_patch)
            self.timing_stats['update_confidence'].append(time.time() - t0)

            i += 1
            if i % to_save_after_each == 0:
                cv2.imwrite(f"tmp/inpainted_image_{i}.jpg", (self.image).astype(np.uint8))
                # Print timing statistics every N iterations
                print(f"\nTiming stats after {i} iterations:")
                for func, times in self.timing_stats.items():
                    avg_time = sum(times[-to_save_after_each:]) / min(to_save_after_each, len(times))
                    print(f"{func}: {avg_time:.4f}s per iteration")

        cv2.imwrite(f"inpainted_image_final.jpg", (self.image).astype(np.uint8))
        end_time = time.time()
        print(f"\nTotal time taken for inpainting: {end_time - start_time:.2f} seconds")
        
        # Print final statistics
        print("\nFinal timing statistics:")
        for func, times in self.timing_stats.items():
            avg_time = sum(times) / len(times)
            total_time = sum(times)
            percentage = (total_time / (end_time - start_time)) * 100
            print(f"{func}:")
            print(f"  Average time: {avg_time:.4f}s")
            print(f"  Total time: {total_time:.2f}s")
            print(f"  Percentage of total: {percentage:.1f}%")