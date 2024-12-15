import React, { useCallback, useState, useRef } from 'react';
import { CssBaseline, Box , CircularProgress, Snackbar, Alert, Dialog, DialogTitle, DialogContent, DialogActions, Button, Select, MenuItem, FormControl, InputLabel, Typography} from '@mui/material';
import OptionsBar from './components/OptionsBar';
import Canvas from './components/Canvas';
import CustomToolbar from './components/CustomToolbar';
import ImageEditorMenuBar from './components/ImageEditorMenuBar';
import LayersPanel from './components/LayersPanel';
import { removeObject, resizeImage, blendImages , inpaintImage} from './services/api';

function App() {
  const canvasRef = useRef(null);
  const [selectedTool, setSelectedTool] = useState(null);
  const [layers, setLayers] = useState([]);
  const [selectedLayerId, setSelectedLayerId] = useState(null);
  const [path, setPath] = useState([]);
  const [brushSize, setBrushSize] = useState(3);
  const [iterations, setIterations] = useState(5);
  const [protectionLayer, setProtectionLayer] = useState(null);
  const [maskLayer, setMaskLayer] = useState(null);
  const [retargetWidth, setRetargetWidth] = useState(0);
  const [retargetHeight, setRetargetHeight] = useState(0);
  const [sourceLayer, setSourceLayer] = useState(null);
  const [targetLayer, setTargetLayer] = useState(null);
  const [blendMode, setBlendMode] = useState('Max');
  const [rightClickedLayerId, setRightClickedLayerId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isForward, setIsForward] = useState(true);
  const [snackbarMessage, setSnackbarMessage] = useState("");
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [addLayerDialogOpen, setAddLayerDialogOpen] = useState(false);
  const [layerType, setLayerType] = useState('image');
  const [parentLayerId, setParentLayerId] = useState(null);

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  const onAddPathOffset = (offset = 0.5) => {
    if(!path.length > 0){
      setSnackbarMessage("There is no selection to add offset to.");
      setSnackbarOpen(true);
      return;
    }

    setPath(prevPath => {
      // Find center point
      const center = prevPath.reduce((acc, point) => ({
        x: acc.x + point.x / prevPath.length,
        y: acc.y + point.y / prevPath.length
      }), { x: 0, y: 0 });
  
      // Add offset in direction from center
      return prevPath.map(point => {
        const dx = point.x - center.x;
        const dy = point.y - center.y;
        const length = Math.sqrt(dx * dx + dy * dy);
        const nx = dx / length; // normalized direction
        const ny = dy / length;
        
        return {
          x: point.x + nx * offset,
          y: point.y + ny * offset
        };
      });
    });
  };

  const onInpaint = useCallback(() => {
    if(!targetLayer){
      setSnackbarMessage("Please select target layer before inpainting.");
      setSnackbarOpen(true);
      return;
    }
    if(!maskLayer){
      setSnackbarMessage("Please select mask layer before inpainting.");
      setSnackbarOpen(true);
      return;
    }
    setIsLoading(true);
    const selectedLayer = targetLayer;
    inpaintImage(selectedLayer.imageUrl, maskLayer.imageUrl)
      .then(res => {
        const img = new Image();
        img.onload = () => {
          const newLayer = {
            id: Date.now(),
            image: img,
            imageUrl: res.data.image,
            visible: true,
            x: 0,
            y: 0,
            scale: 1,
            rotation: 0
          };
          setLayers([...layers, newLayer]);
          setSelectedLayerId(newLayer.id);
          setIsLoading(false);
        };
        img.src = res.data.image;
      })
      .catch(err => {
        setSnackbarMessage("Inpainting failed. Please try again.");
        setSnackbarOpen(true);
        setIsLoading(false);
      });
  }, [layers, maskLayer, targetLayer, setLayers, setSelectedLayerId, setIsLoading]);

  const onHealAI = useCallback(() => {
    console.log("Heal with AI");
    // TODO: Implement AI heal
  }, []);

  const onBlend = useCallback(() => {
    if(!targetLayer){
      setSnackbarMessage("Please select target layer before blending.");
      setSnackbarOpen(true);
      return;
    }
    if(!sourceLayer){
      setSnackbarMessage("Please select source layer before blending.");
      setSnackbarOpen(true);
      return;
    }
    if(!maskLayer){
      setSnackbarMessage("Please select mask layer before blending.");
      setSnackbarOpen(true);
      return;
    }
    setIsLoading(true);
    // Create temporary canvas at the same size as the main canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasRef.current.width;
    tempCanvas.height = canvasRef.current.height;
    const tempCtx = tempCanvas.getContext('2d');
    // Draw the source layer on black background
    tempCtx.clearRect(0, 0, tempCanvas.width, tempCanvas.height);
    tempCtx.save();
    tempCtx.translate(sourceLayer.x, sourceLayer.y);
    tempCtx.rotate(sourceLayer.rotation * Math.PI / 180);
    tempCtx.scale(sourceLayer.scale, sourceLayer.scale);
    tempCtx.drawImage(sourceLayer.image, -sourceLayer.image.width / 2, -sourceLayer.image.height / 2);
    tempCtx.restore();
    // Load the original image to get its dimensions
    const selectedImg = targetLayer.image;
    // Create final mask canvas at original image size
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = selectedImg.width;
    maskCanvas.height = selectedImg.height;
    const maskCtx = maskCanvas.getContext('2d');
    // Calculate the transformation to map back to original image space
    const scaleX = 1 / targetLayer.scale;
    const scaleY = 1 / targetLayer.scale;
    // Clear mask canvas with black
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    // Apply inverse transform and draw the mask
    maskCtx.save();
    maskCtx.translate(selectedImg.width / 2, selectedImg.height / 2);
    maskCtx.rotate(-targetLayer.rotation * Math.PI / 180);
    maskCtx.scale(scaleX, scaleY);
    maskCtx.translate(-targetLayer.x, -targetLayer.y);
    // Draw the temporary canvas content onto the final mask
    maskCtx.drawImage(tempCanvas, 0, 0);
    maskCtx.restore();
    const maskImage = new Image();
    const processedSourceImage = maskCanvas.toDataURL('image/png')
    maskImage.src = processedSourceImage

    // Call API function
    blendImages(
      processedSourceImage,
      maskLayer.imageUrl,
      targetLayer.imageUrl,
      blendMode
    )
      .then(res => {
        const img = new Image();
        img.onload = () => {
          const newLayer = {
            id: Date.now(),
            image: img,
            imageUrl: res.data.image,
            visible: true,
            x: 0,
            y: 0,
            scale: 1,
            rotation: 0
          };
          setLayers([...layers, newLayer]);
          setSelectedLayerId(newLayer.id);
          setIsLoading(false);
        };
        img.src = res.data.image;
      })
      .catch(err => {
        setSnackbarMessage("Blend failed. Please try again.");
        setSnackbarOpen(true);
        setIsLoading(false);
      });

  }, [targetLayer, maskLayer, sourceLayer, setLayers, blendMode, setIsLoading]);

  const onRemove = useCallback(() => {
    if(!targetLayer){
      setSnackbarMessage("Please select target layer before removing.");
      setSnackbarOpen(true);
      return;
    }
    if(!maskLayer){
      setSnackbarMessage("Please select mask layer before removing.");
      setSnackbarOpen(true);
      return;
    }
    setIsLoading(true);
    const selectedLayer = targetLayer;
    removeObject(selectedLayer.imageUrl, maskLayer?.imageUrl, protectionLayer?.imageUrl, isForward)
      .then(res => {
        const img = new Image();
        img.onload = () => {
          const newLayer = {
            id: Date.now(),
            image: img,
            imageUrl: res.data.image,
            visible: true,
            x: 0,
            y: 0,
            scale: 1,
            rotation: 0
          };
          setLayers([...layers, newLayer]);
          setSelectedLayerId(newLayer.id);
          setIsLoading(false);
        };
        img.src = res.data.image;
      })
      .catch(err => {
        setSnackbarMessage("Remove failed. Please try again.");
        setSnackbarOpen(true);
        setIsLoading(false);
      });
  }, [layers, maskLayer, protectionLayer, targetLayer, isForward, setLayers, setSelectedLayerId, setIsLoading]);

  const onRetargetApply = useCallback(() => {
    if(!targetLayer){
      setSnackbarMessage("Please select target layer before retargeting.");
      setSnackbarOpen(true);
      return;
    }
    if(retargetHeight === 0 && retargetWidth === 0){
      setSnackbarMessage("Please enter a valid retarget height and width.");
      setSnackbarOpen(true);
      return;
    }
    
    setIsLoading(true);
    const selectedLayer = targetLayer;
    resizeImage(selectedLayer.imageUrl, selectedLayer.image.height + (retargetHeight * selectedLayer.image.height / 100), selectedLayer.image.width + (retargetWidth * selectedLayer.image.width / 100), protectionLayer?.imageUrl, isForward)
      .then(res => {
        const img = new Image();
        img.onload = () => {
          const newLayer = {
            id: Date.now(),
            image: img,
            imageUrl: res.data.image,
            visible: true,
            x: 0,
            y: 0,
            scale: 1,
            rotation: 0
          };
          setLayers([...layers, newLayer]);
          setSelectedLayerId(newLayer.id);
          setIsLoading(false);
        };
        img.src = res.data.image;
      })
      .catch(err => {
        setSnackbarMessage("Retarget failed. Please try again.");
        setSnackbarOpen(true);
        setIsLoading(false);
      });
  }, [layers, maskLayer, protectionLayer, retargetHeight, retargetWidth, targetLayer, isForward, setLayers, setSelectedLayerId, setIsLoading]);

  const onSelect = useCallback(() => {
    if (canvasRef.current?.onSelect) {
      canvasRef.current.onSelect();
    }
  }, []);

  const onResetIndicators = useCallback(() => {
    if (canvasRef.current?.onResetIndicators) {
      canvasRef.current.onResetIndicators();
    }
  }, []);

  const onRectTool = useCallback(() => {
    if (canvasRef.current?.onRectTool) {
      canvasRef.current.onRectTool();
    }
  }, []);

  const handleCreateMask = useCallback(() => {
    if (canvasRef.current?.handleCreateMask) {
      canvasRef.current.handleCreateMask();
    }
  }, []);

  const handleApplyMask = useCallback((maskLayerId) => {
    const maskLayer = layers.find(layer => layer.id === maskLayerId);
    if (!maskLayer || maskLayer.type !== 'mask') {
      setSnackbarMessage("Selected layer is not a mask.");
      setSnackbarOpen(true);
      return;
    }

    const parentLayer = layers.find(layer => layer.id === maskLayer.parentLayerId);
    if (!parentLayer) {
      setSnackbarMessage("No parent layer found for the mask.");
      setSnackbarOpen(true);
      return;
    }

    if (canvasRef.current?.handleApplyMask) {
      canvasRef.current.handleApplyMask(maskLayer, parentLayer);
    }
  }, [layers]);

  const handleAddLayer = useCallback(() => {
    setAddLayerDialogOpen(true);
  }, []);

  const handleAddLayerConfirm = useCallback(() => {
    if (layerType === 'mask' && layers.length === 0) {
      setSnackbarMessage("No parent layers available to create a mask.");
      setSnackbarOpen(true);
      return;
    }

    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'image/*';
    fileInput.onchange = (e) => {
      const file = e.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onload = (event) => {
          const img = new Image();
          img.onload = () => {
            const newLayer = {
              id: Date.now(),
              image: img,
              imageUrl: event.target.result,
              visible: true,
              x: 0,
              y: 0,
              scale: 1,
              rotation: 0,
              type: layerType,
              parentLayerId: layerType === 'mask' ? parentLayerId : null
            };
            setLayers(prev => [...prev, newLayer]);
            setSelectedLayerId(newLayer.id);
          };
          img.src = event.target.result;
        };
        reader.readAsDataURL(file);
      }
    };
    fileInput.click();
    setAddLayerDialogOpen(false);
  }, [layerType, parentLayerId, layers]);

  const handleToggleVisibility = useCallback((layerId) => {
    setLayers(prev =>
      prev.map(layer =>
        layer.id === layerId ? { ...layer, visible: !layer.visible } : layer
      )
    );
  }, []);

  const handleDeleteLayer = useCallback((layerId) => {
    setLayers(prev => prev.filter(layer => layer.id !== layerId));
    if (selectedLayerId === layerId) {
      setSelectedLayerId(null);
    }
  }, [selectedLayerId]);

     const handleMasksMerge = useCallback(() => {
    // Get all mask layers linked to the right-clicked layer
    const maskLayers = layers.filter(layer => 
      layer.type === 'mask' && 
      layer.parentLayerId === rightClickedLayerId
    );
  
    if (maskLayers.length < 2) {
      setSnackbarMessage("There are less than 2 masks to merge.");
      setSnackbarOpen(true);
      return;
    } // Need at least 2 masks to merge
  
    // Create a temporary canvas for merging
    const mergeCanvas = document.createElement('canvas');
    const firstMask = maskLayers[0].image;
    mergeCanvas.width = firstMask.width;
    mergeCanvas.height = firstMask.height;
    const mergeCtx = mergeCanvas.getContext('2d');
  
    // Draw all masks onto the merge canvas
    maskLayers.forEach(maskLayer => {
      // For each mask, draw it with 'lighter' blend mode to combine white areas
      mergeCtx.globalCompositeOperation = 'lighter';
      mergeCtx.drawImage(
        maskLayer.image,
        0, 0,
        maskLayer.image.width,
        maskLayer.image.height
      );
    });
  
    // Create new merged mask image
    const mergedImage = new Image();
    mergedImage.onload = () => {
      // Create new merged mask layer with properties from first mask
      const mergedMaskLayer = {
        ...maskLayers[0],
        id: Date.now(),
        image: mergedImage,
        imageUrl: mergeCanvas.toDataURL('image/png')
      };
  
      // Remove all original mask layers and add the merged one
      setLayers(prev => [
        ...prev.filter(layer => 
          !(layer.type === 'mask' && layer.parentLayerId === rightClickedLayerId)
        ),
        mergedMaskLayer
      ]);
    };
    mergedImage.src = mergeCanvas.toDataURL('image/png');
  
  }, [layers, rightClickedLayerId]);

  return (
    <>
      <CssBaseline />
      {isLoading && (
      <Box
        position="absolute"
        top={0}
        left={0}
        right={0}
        bottom={0}
        display="flex"
        alignItems="center"
        justifyContent="center"
        bgcolor="rgba(0, 0, 0, 0.5)"
        zIndex={9999}
      >
        <CircularProgress />
      </Box>
    )}
      <ImageEditorMenuBar
        layers={layers}
        selectedLayerId={selectedLayerId}
      />
      <OptionsBar
        selectedTool={selectedTool}
        onCreateMask={handleCreateMask}
        onApplyMask={handleApplyMask}
        onSelect={onSelect}
        onResetIndicators={onResetIndicators}
        onRectTool={onRectTool}
        brushSize={brushSize}
        setBrushSize={setBrushSize}
        iterations={iterations}
        setIterations={setIterations}
        onRemove={onRemove}
        onRetargetApply={onRetargetApply}
        retargetWidth={retargetWidth}
        onRetargetWidthChange={setRetargetWidth}
        retargetHeight={retargetHeight}
        onRetargetHeightChange={setRetargetHeight}
        onBlend={onBlend}
        blendMode={blendMode}
        setBlendMode={setBlendMode}
        setIsForward={setIsForward}
        isForward={isForward}
        onAddPathOffset={onAddPathOffset}
        onHeal={onInpaint}
        onHealAI={onHealAI}
      />
      <Box display="flex" height="calc(100vh - 88px)" bgcolor="background.default">
        <CustomToolbar onSelectTool={setSelectedTool} selectedTool={selectedTool}
        />
        <Box flex={1} display="flex" flexDirection="column">
          <Canvas
            layers={layers}
            selectedLayerId={selectedLayerId}
            selectedTool={selectedTool}
            setLayers={setLayers}
            canvasRef={canvasRef}
            path={path}
            setPath={setPath}
            brushSize={brushSize}
            iterations={iterations}
            setIsLoading={setIsLoading}
          />
        </Box>
        <LayersPanel
          layers={layers}
          selectedLayerId={selectedLayerId}
          onToggleVisibility={handleToggleVisibility}
          onDeleteLayer={handleDeleteLayer}
          onAddLayer={handleAddLayer}
          onSelectLayer={setSelectedLayerId}
          onApplyMask={handleApplyMask}
          protectionLayer={protectionLayer}
          setProtectionLayer={setProtectionLayer}
          maskLayer={maskLayer}
          setMaskLayer={setMaskLayer}
          sourceLayer={sourceLayer}
          setSourceLayer={setSourceLayer}
          targetLayer={targetLayer}
          setTargetLayer={setTargetLayer}
          handleMasksMerge={handleMasksMerge}
          rightClickedLayerId={rightClickedLayerId}
          setRightClickedLayerId={setRightClickedLayerId}
        />
      </Box>
      <Snackbar
        open={snackbarOpen}
        autoHideDuration={6000}
        onClose={handleSnackbarClose}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert onClose={handleSnackbarClose} severity="error" sx={{ width: '100%' }}>
          {snackbarMessage}
        </Alert>
      </Snackbar>
      <Dialog
        open={addLayerDialogOpen}
        onClose={() => setAddLayerDialogOpen(false)}
        maxWidth="sm"
        fullWidth
      >
        <DialogTitle sx={{ bgcolor: '#333', color: '#fff' }}>Add Layer</DialogTitle>
        <DialogContent sx={{ bgcolor: '#2b2b2b', color: '#fff' }}>
          <FormControl fullWidth margin="normal">
            <InputLabel sx={{ color: '#fff' }}>Type</InputLabel>
            <Select
              value={layerType}
              onChange={(e) => setLayerType(e.target.value)}
              sx={{ color: '#fff', '.MuiSelect-icon': { color: '#fff' } }}
            >
              <MenuItem value="image">Image</MenuItem>
              <MenuItem value="mask">Mask</MenuItem>
            </Select>
          </FormControl>
          {layerType === 'mask' && layers.length > 0 && (
            <FormControl fullWidth margin="normal">
              <InputLabel sx={{ color: '#fff' }}>Parent Layer</InputLabel>
              <Select
                value={parentLayerId}
                onChange={(e) => setParentLayerId(e.target.value)}
                sx={{ color: '#fff', '.MuiSelect-icon': { color: '#fff' } }}
              >
                {layers
                  .filter(layer => layer.type !== 'mask')
                  .map((layer, index) => (
                    <MenuItem key={layer.id} value={layer.id}>
                      {`Layer ${index + 1}`}
                    </MenuItem>
                  ))}
              </Select>
            </FormControl>
          )}
          {layerType === 'mask' && layers.length === 0 && (
            <Typography variant="body2" sx={{ color: '#f48fb1', mt: 2 }}>
              No parent layers available to create a mask.
            </Typography>
          )}
        </DialogContent>
        <DialogActions sx={{ bgcolor: '#333' }}>
          <Button onClick={() => setAddLayerDialogOpen(false)} sx={{ color: '#fff' }}>Cancel</Button>
          <Button onClick={handleAddLayerConfirm} color="primary" sx={{ color: '#fff' }}>Add</Button>
        </DialogActions>
      </Dialog>
    </>
  );
}

export default App;