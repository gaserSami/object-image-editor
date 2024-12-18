import React, { useState, useRef, useEffect, useCallback, memo } from 'react';
import { Box, Snackbar, Alert } from '@mui/material';
import { selectObject, refineSelection } from '../services/api';

const handleSize = 8;

const getMousePos = (e, canvasRef) => {
  if (!canvasRef.current) return { x: 0, y: 0 };
  const boundingRect = canvasRef.current.getBoundingClientRect();
  return {
    x: e.clientX - boundingRect.left,
    y: e.clientY - boundingRect.top
  };
};

const isPointInHandle = (clickPoint, handleCenter) => {
  if (!clickPoint || !handleCenter) return false;
  const lowerX = handleCenter.x - handleSize / 2;
  const lowerY = handleCenter.y - handleSize / 2;
  const upperX = handleCenter.x + handleSize / 2;
  const upperY = handleCenter.y + handleSize / 2;
  return (
    clickPoint.x >= lowerX &&
    clickPoint.x <= upperX &&
    clickPoint.y >= lowerY &&
    clickPoint.y <= upperY
  );
};

const isPointInImage = (point, layer) => {
  if (!layer || !point) return false;
  const img = layer.image;
  const dx = point.x - layer.x;
  const dy = point.y - layer.y;
  const angle = -layer.rotation * Math.PI / 180;
  const localX = dx * Math.cos(angle) - dy * Math.sin(angle);
  const localY = dx * Math.sin(angle) + dy * Math.cos(angle);
  const width = img.width * layer.scale;
  const height = img.height * layer.scale;
  return (
    localX >= -width / 2 &&
    localX <= width / 2 &&
    localY >= -height / 2 &&
    localY <= height / 2
  );
};

const drawLayers = (ctx, canvas, layers) => {
  if (!ctx || !canvas || !layers) return;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  layers.forEach(layer => {
    if (!layer.visible || !layer.image.src) return;
    ctx.save();
    ctx.translate(layer.x, layer.y);
    ctx.rotate(layer.rotation * Math.PI / 180);
    ctx.scale(layer.scaleX || layer.scale, layer.scaleY || layer.scale);
    ctx.drawImage(layer.image, -layer.image.width / 2, -layer.image.height / 2);
    ctx.restore();
  });
};

const drawIndicators = (ctx, indicators, brushSize) => {
  if (!ctx || !indicators) return;
  indicators.forEach(indicator => {
    ctx.beginPath();
    ctx.fillStyle = indicator.type === 0 ? 'rgba(0,0,0,1)' : 'rgba(255,255,255,1)';
    ctx.arc(indicator.pos.x, indicator.pos.y, brushSize, 0, Math.PI * 2);
    ctx.fill();
  });
};

const drawLassoPath = (ctx, path) => {
  if (!ctx || path.length === 0) return;
  ctx.save();
  ctx.setLineDash([5, 5]);
  ctx.lineDashOffset = 0;
  ctx.strokeStyle = '#000';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(path[0].x, path[0].y);
  path.forEach(point => ctx.lineTo(point.x, point.y));
  ctx.closePath();
  ctx.stroke();
  ctx.strokeStyle = '#fff';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
};

const getTransformHandles = (layer) => {
  if (!layer || !layer.image.src) return;
  const scaleX = layer.scaleX || layer.scale;
  const scaleY = layer.scaleY || layer.scale;
  const width = layer.image.width * scaleX;
  const height = layer.image.height * scaleY;
  const x = layer.x;
  const y = layer.y;
  return [
    { id: 'nw', x: x - width / 2, y: y - height / 2 },
    { id: 'ne', x: x + width / 2, y: y - height / 2 },
    { id: 'se', x: x + width / 2, y: y + height / 2 },
    { id: 'sw', x: x - width / 2, y: y + height / 2 },
    { id: 'n', x: x, y: y - height / 2 },
    { id: 's', x: x, y: y + height / 2 },
    { id: 'e', x: x + width / 2, y: y },
    { id: 'w', x: x - width / 2, y: y },
    { id: 'rotate', x: x, y: y - height / 2 - 20 }
  ];
};

const drawTransformHandles = (ctx, layer) => {
  if (!ctx || !layer || !layer.image.src || !layer.visible) return;
  const handles = getTransformHandles(layer);
  handles.forEach(handle => {
    ctx.beginPath();
    if (handle.id === 'rotate') {
      ctx.arc(handle.x, handle.y, handleSize / 2, 0, Math.PI * 2);
    } else {
      ctx.rect(handle.x - handleSize / 2, handle.y - handleSize / 2, handleSize, handleSize);
    }
    ctx.fillStyle = '#83bff7';
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 1;
    ctx.fill();
    ctx.stroke();
  });
};

const extractBoundaryPathFromMask = (maskLayer) => {
  // Create temporary canvas for mask processing
  const maskCanvas = document.createElement('canvas');
  const maskCtx = maskCanvas.getContext('2d');
  maskCanvas.width = maskLayer.image.width;
  maskCanvas.height = maskLayer.image.height;
  maskCtx.drawImage(maskLayer.image, 0, 0);

  const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
  const data = maskData.data;
  
  // Helper function to check if pixel is white (part of mask)
  const isWhitePixel = (x, y) => {
    if (x < 0 || y < 0 || x >= maskCanvas.width || y >= maskCanvas.height) return false;
    const idx = (y * maskCanvas.width + x) * 4;
    return data[idx] > 128 || data[idx + 1] > 128 || data[idx + 2] > 128;
  };

  // Find the boundary pixels
  const boundaryPoints = [];
  for (let y = 0; y < maskCanvas.height; y++) {
    for (let x = 0; x < maskCanvas.width; x++) {
      if (isWhitePixel(x, y)) {
        // Check if this is a boundary pixel by looking at neighbors
        const isBoundary = !isWhitePixel(x-1, y) || !isWhitePixel(x+1, y) ||
                          !isWhitePixel(x, y-1) || !isWhitePixel(x, y+1);
        if (isBoundary) {
          boundaryPoints.push({ x, y });
        }
      }
    }
  }

  if (boundaryPoints.length === 0) return [];

  // Sort points to create a continuous path
  const path = [boundaryPoints[0]];
  const used = new Set([`${boundaryPoints[0].x},${boundaryPoints[0].y}`]);
  
  while (path.length < boundaryPoints.length) {
    const last = path[path.length - 1];
    let closest = null;
    let minDist = Infinity;
    
    // Find the closest unused point
    for (const point of boundaryPoints) {
      const key = `${point.x},${point.y}`;
      if (!used.has(key)) {
        const dist = Math.hypot(point.x - last.x, point.y - last.y);
        if (dist < minDist) {
          minDist = dist;
          closest = point;
        }
      }
    }
    
    if (!closest || minDist > 2) break; // Stop if no close points found
    path.push(closest);
    used.add(`${closest.x},${closest.y}`);
  }

  // Transform points to canvas space
  const transformedPath = path.map(point => {
    // Convert from image space to normalized space (-0.5 to 0.5)
    const normalizedX = (point.x / maskLayer.image.width) - 0.5;
    const normalizedY = (point.y / maskLayer.image.height) - 0.5;

    // Apply scale
    const scaledX = normalizedX * maskLayer.image.width * maskLayer.scale;
    const scaledY = normalizedY * maskLayer.image.height * maskLayer.scale;

    // Apply rotation
    const angle = maskLayer.rotation * Math.PI / 180;
    const rotatedX = scaledX * Math.cos(angle) - scaledY * Math.sin(angle);
    const rotatedY = scaledX * Math.sin(angle) + scaledY * Math.cos(angle);

    // Apply translation
    return {
      x: rotatedX + maskLayer.x,
      y: rotatedY + maskLayer.y
    };
  });

  // Smooth the path using Chaikin's algorithm
  const smoothPath = (points, iterations = 2) => {
    if (points.length < 3) return points;
    
    for (let iter = 0; iter < iterations; iter++) {
      const newPoints = [];
      for (let i = 0; i < points.length; i++) {
        const p0 = points[i];
        const p1 = points[(i + 1) % points.length];
        
        newPoints.push({
          x: 0.75 * p0.x + 0.25 * p1.x,
          y: 0.75 * p0.y + 0.25 * p1.y
        });
        
        newPoints.push({
          x: 0.25 * p0.x + 0.75 * p1.x,
          y: 0.25 * p0.y + 0.75 * p1.y
        });
      }
      points = newPoints;
    }
    return points;
  };

  // Return smoothed path
  return smoothPath(transformedPath);
};

const Canvas = memo(function Canvas({
  layers,
  selectedLayerId,
  selectedTool,
  setLayers,
  canvasRef,
  path,
  setPath,
  brushSize,
  iterations,
  setIsLoading,
}) {
  const contextRef = useRef(null);
  const [isDragging, setIsDragging] = useState(false);
  const [startPos, setStartPos] = useState({ x: 0, y: 0 });
  const [transformHandle, setTransformHandle] = useState(null);
  const [isLassoDrawing, setIsLassoDrawing] = useState(false);

  const [rectStart, setRectStart] = useState(null);
  const [rectEnd, setRectEnd] = useState(null);
  const [isDrawingRect, setIsDrawingRect] = useState(false);
  const [mask2dArr, setMask2dArr] = useState(null);
  const [indicators, setIndicators] = useState([]);
  const [isDrawingIndicators, setIsDrawingIndicators] = useState(false);
  const [drawingType, setDrawingType] = useState(null); // null, 0, or 1 for drawing type
  const [isDrawing, setIsDrawing] = useState(false);
  const [snackbarOpen, setSnackbarOpen] = useState(false);
  const [snackbarMessage, setSnackbarMessage] = useState('');

  const onRectTool = useCallback(() => {
    setIsDrawingRect(true);
  }, []);

  const onSelect = useCallback(async () => {
    if (!selectedLayerId) {
      setSnackbarMessage("Cannot select: No selected layer");
      setSnackbarOpen(true);
      return;
    }

    const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
    if (!selectedLayer) {
      setSnackbarMessage("Cannot select: No selected layer. There was issue while getting the layer.");
      setSnackbarOpen(true);
      return;
    }

    if(!rectStart && !path.length > 0){
      setSnackbarMessage("Cannot enhance selection. Please use Intial Selection tool to create a selection first.");
      setSnackbarOpen(true);
      return;
    }

    let rect;
    let img = selectedLayer.image;
    let angle = -selectedLayer.rotation * Math.PI / 180;
    let scale = selectedLayer.scale;

    if (rectStart && rectEnd) {
      // Convert rect coordinates to image space
      let x = Math.min(rectStart.x, rectEnd.x);
      let y = Math.min(rectStart.y, rectEnd.y);
      let width = Math.abs(rectEnd.x - rectStart.x);
      let height = Math.abs(rectEnd.y - rectStart.y);

      // Translate to layer's coordinate system center
      const dx = selectedLayer.x;
      const dy = selectedLayer.y;
      const localX = x - dx;
      const localY = y - dy;

      // Apply inverse rotation
      const rotatedX = localX * Math.cos(angle) - localY * Math.sin(angle);
      const rotatedY = localX * Math.sin(angle) + localY * Math.cos(angle);

      // Scale to original image dimensions and translate to image center
      x = (rotatedX / scale) + (img.width / 2);
      y = (rotatedY / scale) + (img.height / 2);
      width = width / scale;
      height = height / scale;

      // Create rectangle in image coordinates
      rect = {
        "x": Math.round(x),
        "y": Math.round(y),
        "width": Math.round(width),
        "height": Math.round(height)
      };
    }

    try {
      let response;
      setIsLoading(true);
      if (path.length > 0) {
        response = await refineSelection(mask2dArr, iterations);
      } else {
        response = await selectObject(selectedLayer.imageUrl, rect, iterations);
      }

      const tempPath = response.data["path"].map(point => ({ x: point[0], y: point[1] }));
      const maskArray = response.data["mask"];
      setMask2dArr(maskArray);
      setIsLoading(false);

      // apply transformation to the path to go back to the original image space
      if(img){
        tempPath.forEach(point => {
          const dx = point.x - img.width / 2;
          const dy = point.y - img.height / 2;
          const rotatedX = dx * Math.cos(angle) - dy * Math.sin(angle);
          const rotatedY = dx * Math.sin(angle) + dy * Math.cos(angle);
          const scaledX = rotatedX * scale + selectedLayer.x;
          const scaledY = rotatedY * scale + selectedLayer.y;
          point.x = scaledX;
          point.y = scaledY;
        });
      }

      setPath(tempPath);

      setRectStart(null);
      setRectEnd(null);
      setIsDrawingRect(false);
    } catch (error) {
      setSnackbarMessage("Selection failed. Please try again.");
      setSnackbarOpen(true);
      setIsLoading(false);
    }
  }, [selectedLayerId, rectStart, rectEnd, layers, iterations, mask2dArr, path, setPath]);

  const onResetIndicators = useCallback(() => {
    if(indicators.length > 0){
      setIndicators([]);
    }else{
      setSnackbarMessage("There are no indicators to reset.");
      setSnackbarOpen(true);
    }
  }, [indicators]);

  const handleSnackbarClose = () => {
    setSnackbarOpen(false);
  };

  const handleCreateMask = useCallback(() => {
    if (!selectedLayerId) {
      setSnackbarMessage("Cannot create mask: No selected layer");
      setSnackbarOpen(true);
      return;
    }
    const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
    if (!selectedLayer) {
      setSnackbarMessage("Cannot create mask: No selected layer");
      setSnackbarOpen(true);
      return;
    }

    // Create temporary canvas at the same size as the main canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = canvasRef.current.width;
    tempCanvas.height = canvasRef.current.height;
    const tempCtx = tempCanvas.getContext('2d');
    
    // Fill with black by default
    tempCtx.fillStyle = 'black';
    tempCtx.fillRect(0, 0, tempCanvas.width, tempCanvas.height);

    // Only draw the white path if it exists
    if (path && path.length > 0) {
      tempCtx.fillStyle = 'white';
      tempCtx.beginPath();
      tempCtx.moveTo(path[0].x, path[0].y);
      path.forEach(point => tempCtx.lineTo(point.x, point.y));
      tempCtx.closePath();
      tempCtx.fill();
    }

    // Rest of the function remains the same...
    const selectedImg = selectedLayer.image;
    const maskCanvas = document.createElement('canvas');
    maskCanvas.width = selectedImg.width;
    maskCanvas.height = selectedImg.height;
    const maskCtx = maskCanvas.getContext('2d');
    const scaleX = 1 / selectedLayer.scale;
    const scaleY = 1 / selectedLayer.scale;
    maskCtx.fillStyle = 'black';
    maskCtx.fillRect(0, 0, maskCanvas.width, maskCanvas.height);
    maskCtx.save();
    maskCtx.translate(selectedImg.width / 2, selectedImg.height / 2);
    maskCtx.rotate(-selectedLayer.rotation * Math.PI / 180);
    maskCtx.scale(scaleX, scaleY);
    maskCtx.translate(-selectedLayer.x, -selectedLayer.y);
    maskCtx.drawImage(tempCanvas, 0, 0);
    maskCtx.restore();
    const maskImage = new Image();
    maskImage.onload = () => {
      setLayers(prev => [...prev, {
        id: Date.now(),
        type: 'mask',
        parentLayerId: selectedLayerId,
        image: maskImage,
        imageUrl: maskCanvas.toDataURL('image/png'),
        visible: true,
        x: selectedLayer.x,
        y: selectedLayer.y,
        scale: selectedLayer.scale,
        rotation: selectedLayer.rotation
      }]);
    };
    maskImage.src = maskCanvas.toDataURL('image/png');
  }, [selectedLayerId, path, layers, canvasRef, setLayers]);

  const handleApplyMask = useCallback(async (maskLayer, parentLayer) => {
    if (!parentLayer || !maskLayer) return;
    const originalCanvas = document.createElement('canvas');
    const originalCtx = originalCanvas.getContext('2d');
    const maskCanvas = document.createElement('canvas');
    const maskCtx = maskCanvas.getContext('2d');
    try {
      const parentImg = parentLayer.image;
      const maskImg = maskLayer.image;
      // Set canvas sizes
      originalCanvas.width = parentImg.width;
      originalCanvas.height = parentImg.height;
      maskCanvas.width = parentImg.width;
      maskCanvas.height = parentImg.height;
      // Draw parent image
      originalCtx.save();
      originalCtx.translate(originalCanvas.width / 2, originalCanvas.height / 2);
      originalCtx.rotate(parentLayer.rotation * Math.PI / 180);
      originalCtx.scale(parentLayer.scale, parentLayer.scale);
      originalCtx.drawImage(
        parentImg,
        -parentImg.width / 2,
        -parentImg.height / 2,
        parentImg.width,
        parentImg.height
      );
      originalCtx.restore();
      // Draw mask
      maskCtx.save();
      maskCtx.translate(maskCanvas.width / 2, maskCanvas.height / 2);
      maskCtx.rotate(maskLayer.rotation * Math.PI / 180);
      maskCtx.scale(maskLayer.scale, maskLayer.scale);
      maskCtx.drawImage(
        maskImg,
        -maskImg.width / 2,
        -maskImg.height / 2,
        maskImg.width,
        maskImg.height
      );
      maskCtx.restore();
      // Process mask pixels
      const maskData = maskCtx.getImageData(0, 0, maskCanvas.width, maskCanvas.height);
      const data = maskData.data;
      for (let i = 0; i < data.length; i += 4) {
        // Check if pixel is black (or very dark)
        const isBlack = data[i] < 128 && data[i + 1] < 128 && data[i + 2] < 128;
        // Set RGB to white
        data[i] = data[i + 1] = data[i + 2] = 255;
        // Set alpha based on black/white
        data[i + 3] = isBlack ? 0 : 255;
      }
      maskCtx.putImageData(maskData, 0, 0);
      // Create final canvas
      const finalCanvas = document.createElement('canvas');
      finalCanvas.width = parentImg.width;
      finalCanvas.height = parentImg.height;
      const finalCtx = finalCanvas.getContext('2d');
      // Draw original image
      finalCtx.drawImage(originalCanvas, 0, 0);
      // Apply mask
      finalCtx.globalCompositeOperation = 'destination-in';
      finalCtx.drawImage(maskCanvas, 0, 0);
      // Find bounds of visible content
      const imageData = finalCtx.getImageData(0, 0, finalCanvas.width, finalCanvas.height);
      const bounds = {
        left: finalCanvas.width,
        right: 0,
        top: finalCanvas.height,
        bottom: 0
      };

      // Scan for non-transparent pixels
      for (let y = 0; y < finalCanvas.height; y++) {
        for (let x = 0; x < finalCanvas.width; x++) {
          const idx = (y * finalCanvas.width + x) * 4;
          const alpha = imageData.data[idx + 3];
          if (alpha > 0) {
            bounds.left = Math.min(bounds.left, x);
            bounds.right = Math.max(bounds.right, x);
            bounds.top = Math.min(bounds.top, y);
            bounds.bottom = Math.max(bounds.bottom, y);
          }
        }
      }

      // Add small padding
      const padding = 2;
      bounds.left = Math.max(0, bounds.left - padding);
      bounds.top = Math.max(0, bounds.top - padding);
      bounds.right = Math.min(finalCanvas.width, bounds.right + padding);
      bounds.bottom = Math.min(finalCanvas.height, bounds.bottom + padding);

      // Create cropped canvas
      const croppedCanvas = document.createElement('canvas');
      croppedCanvas.width = bounds.right - bounds.left;
      croppedCanvas.height = bounds.bottom - bounds.top;
      const croppedCtx = croppedCanvas.getContext('2d');

      // Copy the visible portion
      croppedCtx.drawImage(
        finalCanvas,
        bounds.left, bounds.top,
        bounds.right - bounds.left, bounds.bottom - bounds.top,
        0, 0,
        bounds.right - bounds.left, bounds.bottom - bounds.top
      );

      const finalCanvasImage = new Image();
      finalCanvasImage.onload = () => {
        const newLayer = {
          id: Date.now(),
          imageUrl: croppedCanvas.toDataURL('image/png'),
          image: finalCanvasImage,
          visible: true,
          x: parentLayer.x + bounds.left,  // Adjust position to account for cropping
          y: parentLayer.y + bounds.top,   // Adjust position to account for cropping
          scale: parentLayer.scale,
          rotation: parentLayer.rotation
        };
        setLayers(prev => {
          const filteredLayers = prev.filter(layer => layer.id !== maskLayer.id);
          return [...filteredLayers, newLayer];
        });
      };
      finalCanvasImage.src = croppedCanvas.toDataURL('image/png');

    } catch (error) {
      console.error('Error applying mask:', error);
    }
  }, [setLayers]);

  const setupCanvas = useCallback(() => {
    const canvas = canvasRef.current;
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientHeight;
    canvas.style.width = `${container.clientWidth}px`;
    canvas.style.height = `${container.clientHeight}px`;
    contextRef.current = canvas.getContext('2d');
  }, [canvasRef]);

  const handleKeyDown = useCallback((e) => {
    console.log(e.key);
    console.log(selectedLayerId);
    if (e.key === 'Escape') {
      setIsDrawingRect(false);
      setRectStart(null);
      setRectEnd(null);
      setPath([]);
    } else if ((e.key === '0' || e.key === '1')) {
      console.log('key down');
      setDrawingType(parseInt(e.key, 10)); // Set drawing type as 0 or 1
      setIsDrawingIndicators(true);
    } else if (e.key === '3') {
      setDrawingType(null);
      setIsDrawingIndicators(false);
    } else if (e.key === 's' && selectedLayerId) {
      const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
      if (selectedLayer && selectedLayer.type === 'mask') {
        console.log('s key down');
        const selectionPath = extractBoundaryPathFromMask(selectedLayer);
        console.log(selectionPath);
        setPath(selectionPath);
      }
    }
  }, [selectedLayerId, layers, setPath]);
  
  useEffect(() => {
    setupCanvas();
    window.addEventListener('resize', setupCanvas);
    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('resize', setupCanvas);
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [handleKeyDown]);

  useEffect(() => {
    canvasRef.current.handleApplyMask = handleApplyMask;
    canvasRef.current.handleCreateMask = handleCreateMask;
    canvasRef.current.onRectTool = onRectTool;
    canvasRef.current.onSelect = onSelect;
    canvasRef.current.onResetIndicators = onResetIndicators;
  }, [handleApplyMask, handleCreateMask, onRectTool, onSelect, onResetIndicators]);

  useEffect(() => {
    drawLayers(contextRef.current, canvasRef.current, layers);
    drawIndicators(contextRef.current, indicators, brushSize);

    if (selectedTool === 'pointer' && selectedLayerId) {
      const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
      drawTransformHandles(contextRef.current, selectedLayer);
    }

    // Ensure the path is drawn consistently
    if (path.length > 0) {
      drawLassoPath(contextRef.current, path);
    }
  }, [layers, selectedLayerId, selectedTool, path, mask2dArr, indicators]);

  const handleMouseDown = useCallback((e) => {
    const pos = getMousePos(e, canvasRef);
    setStartPos(pos);

    if (isDrawingIndicators) {
      setIsDrawing(true);
      return;
    }

    if (isDrawingRect) {
      setRectStart(pos);
      setRectEnd(pos);
      return;
    }

    if (selectedTool === 'lasso') {
      setPath([pos]);
      setIsLassoDrawing(true);
    } else if (selectedLayerId) {
      const selectedLayer = layers.find(l => l.id === selectedLayerId);
      const handle = getTransformHandles(selectedLayer, selectedLayer.image)
        .find(h => isPointInHandle(pos, h));
      if (handle) {
        setTransformHandle(handle.id);
      } else if (isPointInImage(pos, selectedLayer) && selectedTool === 'pointer') {
        setIsDragging(true);
      }
    }
  }, [selectedTool, selectedLayerId, layers, isDrawingRect, isDrawingIndicators, mask2dArr]);

  const handleMouseMove = useCallback((e) => {
    const pos = getMousePos(e, canvasRef);

    if (isDrawingRect && rectStart) {
      setRectEnd(pos);
      const ctx = contextRef.current;
      drawLayers(ctx, canvasRef.current, layers);
      // Draw rectangle
      ctx.beginPath();
      ctx.strokeStyle = '#fff';
      ctx.lineWidth = 2;
      ctx.strokeRect(
        rectStart.x,
        rectStart.y,
        pos.x - rectStart.x,
        pos.y - rectStart.y
      );
      return;
    }

    if (isDrawingIndicators && isDrawing) {
      console.log('drawing');
      // Draw indicator preview while moving
      const ctx = contextRef.current;
      // Draw the brush cursor as a preview with the appropriate color
      ctx.beginPath();
      ctx.fillStyle = drawingType === 0 ? 'rgba(0,0,0,0.5)' : 'rgba(255,255,255,0.5)';
      ctx.arc(pos.x, pos.y, brushSize, 0, Math.PI * 2);
      ctx.fill();
      setIndicators(prev => [...prev, { pos, type: drawingType }]);
      updateMask2dArr(pos, drawingType);
      return;
    }

    if (selectedTool === 'lasso' && isLassoDrawing) {
      setPath(prev => [...prev, pos]);
      const ctx = contextRef.current;
      drawLayers(ctx, canvasRef.current, layers);
      drawLassoPath(ctx, [...path, pos]);
      return;
    }

    if (!selectedLayerId || (!isDragging && !transformHandle)) return;
    const selectedLayer = layers.find(l => l.id === selectedLayerId);
    if (isDragging) {
      const dx = pos.x - startPos.x;
      const dy = pos.y - startPos.y;
      setLayers(prev => prev.map(layer =>
        layer.id === selectedLayerId
          ? { ...layer, x: layer.x + dx, y: layer.y + dy }
          : layer
      ));
    } else if (transformHandle) {
      const dx = pos.x - startPos.x;
      const dy = pos.y - startPos.y;
      if (transformHandle === 'rotate') {
        const angle = Math.atan2(pos.y - selectedLayer.y, pos.x - selectedLayer.x);
        setLayers(prev => prev.map(layer =>
          layer.id === selectedLayerId
            ? { ...layer, rotation: angle * (180 / Math.PI) }
            : layer
        ));
      } else if (['n', 's'].includes(transformHandle)) {
        // Vertical scaling
        const scaleY = Math.max(0.1, (selectedLayer.scaleY || selectedLayer.scale) * (1 + dy / 100));
        setLayers(prev => prev.map(layer =>
          layer.id === selectedLayerId
            ? { ...layer, scaleY }
            : layer
        ));
      } else if (['e', 'w'].includes(transformHandle)) {
        // Horizontal scaling
        const scaleX = Math.max(0.1, (selectedLayer.scaleX || selectedLayer.scale) * (1 + dx / 100));
        setLayers(prev => prev.map(layer =>
          layer.id === selectedLayerId
            ? { ...layer, scaleX }
            : layer
        ));
      } else {
        // Uniform scaling for corner handles
        const scale = Math.max(0.1, selectedLayer.scale * (1 + (dx + dy) / 100));
        setLayers(prev => prev.map(layer =>
          layer.id === selectedLayerId
            ? { ...layer, scale, scaleX: scale, scaleY: scale }
            : layer
        ));
      }
    }
    setStartPos(pos);
    // drawLayers(contextRef.current, canvasRef.current, layers);
  }, [selectedTool, selectedLayerId, layers, isDragging, transformHandle, path, isDrawingRect, rectStart, isDrawingIndicators, mask2dArr, brushSize, indicators, isDrawingIndicators, isDrawing]);

  const handleMouseUp = useCallback(() => {
    if (selectedTool === 'lasso' && isLassoDrawing) {
      setIsLassoDrawing(false);
      if (path.length > 2) {
        const firstPoint = path[0];
        const lastPoint = path[path.length - 1];
        if (Math.hypot(lastPoint.x - firstPoint.x, lastPoint.y - firstPoint.y) <= 20) {
          setPath(prev => [...prev, firstPoint]);
        }
      }
    }
    setIsDrawing(false);
    setIsDragging(false);
    setTransformHandle(null);
    setIsDrawingRect(false);
  }, [selectedTool, path]);

    const updateMask2dArr = useCallback((canvasPos, type) => {
      if (!mask2dArr || !layers) return;
  
      const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
      if (!selectedLayer) return;
  
      // 1. Transform from canvas to layer-local coordinates
      const dx = canvasPos.x - selectedLayer.x;
      const dy = canvasPos.y - selectedLayer.y;
      
      // 2. Apply rotation transform
      const angle = -selectedLayer.rotation * Math.PI / 180;
      const localX = dx * Math.cos(angle) - dy * Math.sin(angle);
      const localY = dx * Math.sin(angle) + dy * Math.cos(angle);
  
      // 3. Convert to image/mask space coordinates
      const imageX = Math.round((localX / selectedLayer.scale) + (selectedLayer.image.width / 2));
      const imageY = Math.round((localY / selectedLayer.scale) + (selectedLayer.image.height / 2));
  
      // Create a copy of the mask array
      const newMask = [...mask2dArr];
      
      // Apply brush in mask space coordinates
      const brushRadius = brushSize;
      for (let dy = -brushRadius; dy <= brushRadius; dy++) {
          for (let dx = -brushRadius; dx <= brushRadius; dx++) {
              const maskX = imageX + dx;
              const maskY = imageY + dy;
              const distance = Math.sqrt(dx * dx + dy * dy);
  
              if (distance <= brushRadius && 
                  maskX >= 0 && maskX < newMask[0].length && 
                  maskY >= 0 && maskY < newMask.length) {
                  newMask[maskY][maskX] = type === 0 ? 0 : 1;
              }
          }
      }
  
      setMask2dArr(newMask);
  }, [mask2dArr, brushSize, selectedLayerId, layers]);

  return (
    <Box sx={{ flex: 1, position: 'relative', overflow: 'hidden', bgcolor: '#1e1e1e' }}>
      <canvas
        ref={canvasRef}
        onMouseDown={handleMouseDown}
        onMouseMove={handleMouseMove}
        onMouseUp={handleMouseUp}
        style={{
          cursor: isDrawingIndicators ? 'default' : selectedTool === 'pointer' ? 'move' : 'crosshair'
        }}
      />
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
    </Box>
  );
});

export default Canvas;