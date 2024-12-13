import axios from 'axios';

// Base URL for the API
const API_URL = 'http://localhost:5000';

/**
 * Select an object in the image.
 * @param {string} image - The image data.
 * @param {object} rect - The rectangle defining the object.
 * @param {number} iter - The iteration number.
 * @returns {Promise} - The axios response promise.
 */
export const selectObject = (image, rect, iter) => {
  return axios.post(`${API_URL}/select-object`, {
    "image": image,
    "rect": rect,
    "iter": iter
  });
};

/**
 * Refine the selection mask.
 * @param {string} mask - The mask data.
 * @param {number} iter - The iteration number.
 * @returns {Promise} - The axios response promise.
 */
export const refineSelection = (mask, iter) => {
  return axios.post(`${API_URL}/refine-selection`, {
    "mask": mask,
    "iter" : iter
  });
};

/**
 * Resize the image.
 * @param {string} image - The image data.
 * @param {number} height - The new height.
 * @param {number} width - The new width.
 * @param {object|null} protection - The protection data (optional).
 * @returns {Promise} - The axios response promise.
 */
export const resizeImage = (image, height, width, protection = null, forward=true) => {
  return axios.post(`${API_URL}/resize-image`, {
    "image": image,
    "height": height,
    "width": width,
    "protection" : protection,
    "forward": forward
  });
};

/**
 * Remove an object from the image.
 * @param {string} image - The image data.
 * @param {string|null} mask - The mask data (optional).
 * @param {object|null} protection - The protection data (optional).
 * @returns {Promise} - The axios response promise.
 */
export const removeObject = (image, mask=null, protection=null, forward=true) => {
  return axios.post(`${API_URL}/remove-object`, {
    "image": image,
    "mask": mask,
    "protection": protection,
    "forward": forward
  });
};

/**
 * Blend two images.
 * @param {string} source - The source image data.
 * @param {string} mask - The mask data.
 * @param {string} target - The target image data.
 * @param {string} method - The blending method (default is 'mix').
 * @returns {Promise} - The axios response promise.
 */
export const blendImages = (source, mask, target, mood = 'Max') => {
  return axios.post(`${API_URL}/blend-images`, {
    "source": source,
    "mask": mask, 
    "target": target,
    "mood": mood // Max, Average, Replace, Sum
  });
};

/**
 * Inpaint an image using exemplar-based method.
 * @param {string} image - The image data.
 * @param {string} mask - The mask data.
 * @param {number} patchSize - The patch size (optional, default 9).
 * @returns {Promise} - The axios response promise.
 */
export const inpaintImage = (image, mask, patchSize = 9) => {
  return axios.post(`${API_URL}/inpaint-image`, {
    "image": image,
    "mask": mask,
    "patch_size": patchSize
  });
};