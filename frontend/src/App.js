import React, { useCallback, useState, useRef } from 'react';
import { CssBaseline, Box , CircularProgress} from '@mui/material';
import OptionsBar from './components/OptionsBar';
import Canvas from './components/Canvas';
import CustomToolbar from './components/CustomToolbar';
import ImageEditorMenuBar from './components/ImageEditorMenuBar';
import LayersPanel from './components/LayersPanel';
import { removeObject, resizeImage, blendImages } from './services/api';

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
  const [blendMode, setBlendMode] = useState('mix');
  const [rightClickedLayerId, setRightClickedLayerId] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  const onBlend = useCallback(() => {
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
        console.error('Blend failed:', err);
        setIsLoading(false);
      });

  }, [targetLayer, maskLayer, sourceLayer, setLayers, blendMode]);

  const onRemove = useCallback(() => {
    setIsLoading(true);
    const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
    removeObject(selectedLayer.imageUrl, maskLayer?.imageUrl, protectionLayer?.imageUrl)
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
        console.error(err);
        setIsLoading(false);
      });
  }, [layers, maskLayer, protectionLayer, selectedLayerId]);

  const onRetargetApply = useCallback(() => {
    setIsLoading(true);
    const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
    resizeImage(selectedLayer.imageUrl, selectedLayer.image.height + (retargetHeight * selectedLayer.image.height / 100), selectedLayer.image.width + (retargetWidth * selectedLayer.image.width / 100), protectionLayer?.imageUrl)
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
        console.error(err);
        setIsLoading(false);
      });
  }, [layers, maskLayer, protectionLayer, retargetHeight, retargetWidth]);

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
    if (!maskLayer || maskLayer.type !== 'mask') return;

    const parentLayer = layers.find(layer => layer.id === maskLayer.parentLayerId);
    if (!parentLayer) return;

    if (canvasRef.current?.handleApplyMask) {
      canvasRef.current.handleApplyMask(maskLayer, parentLayer);
    }
  }, [layers]);

  const handleAddLayer = useCallback(() => {
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
              rotation: 0
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
  }, []);

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
  
    if (maskLayers.length < 2) return; // Need at least 2 masks to merge
  
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
    </>
  );
}

export default App;