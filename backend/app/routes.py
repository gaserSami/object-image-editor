from flask import request, jsonify
from app.services.image_service import ImageService
from app.services.seam_craver_service import SeamCarverService
from app.services.poisson_service import PoissonService
from app.utils.image_utils import ImageUtils
from app import app
import traceback
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

@app.route('/select-object', methods=['POST'])
def select_object():
    """
    Endpoint to start object selection in an image.
    
    Request JSON:
    {
        "image": <image_data>,
        "rect": <selection_rectangle>,
        "iter": <number_of_iterations>
    }
    
    Response JSON:
    {
        "mask": <selection_mask>,
        "path": <path_to_saved_image>
    }
    """
    data = request.json
        
    try:
        mask, path = ImageService.start_selection(
            image_data=data['image'],
            rect=data['rect'],
            iter=data['iter'],
        )
        return jsonify({
            "mask": mask,
            "path": path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route("/refine-selection", methods=['POST'])
def refine_selection():
    """
    Endpoint to refine object selection in an image.
    
    Request JSON:
    {
        "mask": <current_selection_mask>,
        "iter": <number_of_iterations>
    }
    
    Response JSON:
    {
        "mask": <refined_selection_mask>,
        "path": <path_to_saved_image>
    }
    """
    data = request.json
        
    try:
        mask, path = ImageService.refine_selection(
            mask=data['mask'],
            iter=data['iter'],
        )
        return jsonify({
            "mask": mask,
            "path": path
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/resize-image', methods=['POST'])
def resize_image():
    """
    Endpoint to resize an image using seam carving with a protection mask.
    
    Request JSON:
    {
        "image": <image_data>,
        "height": <new_height>,
        "width": <new_width>,
        "protection": <protection_mask>
    }
    
    Response JSON:
    {
        "image": <resized_image_data>
    }
    """
    data = request.json
    
    try:
        result = SeamCarverService.resize_with_mask(
            image_data=data['image'],
            new_height=data['height'],
            new_width=data['width'],
            protect_mask=data['protection'],
        )
        return jsonify({
                "image": ImageUtils.encode_image(result)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/remove-object', methods=['POST'])
def remove_object():
    """
    Endpoint to remove an object from an image using seam carving.
    
    Request JSON:
    {
        "image": <image_data>,
        "mask": <object_mask>,
        "protection": <protection_mask>
    }
    
    Response JSON:
    {
        "image": <image_data_with_object_removed>
    }
    """
    data = request.json
    
    try:
        result = SeamCarverService.remove_object(
            image_data=data['image'],
            object_mask=data['mask'],
            protect_mask=data['protection']
        )
        
        return jsonify({
            "image": ImageUtils.encode_image(result)
        })
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in remove_object: {str(e)}\n{error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500

@app.route('/blend-images', methods=['POST'])
def blend_images():
    """
    Endpoint to blend two images using Poisson image editing.
    
    Request JSON:
    {
        "source": <source_image_data>,
        "mask": <mask_data>,
        "target": <target_image_data>,
        "method": <blending_method> (optional, default is 'import')
    }
    
    Response JSON:
    {
        "image": <blended_image_data>
    }
    """
    data = request.json
    
    try:
        result = PoissonService.blend_images(
            source_data=data['source'],
            mask_data=data['mask'],
            target_data=data['target'],
            mood=data.get('mood', 'Mix')
        )
        
        return jsonify({
            "image": ImageUtils.encode_image(result)
        })
    
    except Exception as e:
        error_trace = traceback.format_exc()
        logger.error(f"Error in blend_images: {str(e)}\n{error_trace}")
        return jsonify({
            'error': str(e),
            'traceback': error_trace
        }), 500