import React, {memo, useCallback, useState} from 'react';
import { Box, List, ListItem, ListItemText, ListItemIcon, IconButton, Typography , Menu, MenuItem} from '@mui/material';
import VisibilityIcon from '@mui/icons-material/Visibility';
import VisibilityOffIcon from '@mui/icons-material/VisibilityOff';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import ContentPasteOffIcon from '@mui/icons-material/ContentPasteOff';

const LayersPanel =  memo(function LayersPanel({ 
  layers, 
  onToggleVisibility, 
  onDeleteLayer, 
  onAddLayer, 
  onSelectLayer, 
  selectedLayerId,
  onApplyMask ,
  protectionLayer,
  setProtectionLayer,
  maskLayer,
  setMaskLayer,
  sourceLayer,
  setSourceLayer,
  targetLayer,
  setTargetLayer,
  handleMasksMerge,
  rightClickedLayerId,
  setRightClickedLayerId
}) {
  const [contextMenu, setContextMenu] = useState(null);

  const rightClickedLayer = layers.find(l => l.id === rightClickedLayerId);

  const getMaskName = useCallback((maskLayer) => {
    const parentLayer = layers.find(l => l.id === maskLayer.parentLayerId);
    return `Mask (${parentLayer ? `Layer ${layers.indexOf(parentLayer) + 1}` : 'Unlinked'})`;
  }, [layers, maskLayer]);

  const handleContextMenu = (event, layerId) => {
    event.preventDefault();
    setRightClickedLayerId(layerId);
    setContextMenu({
      mouseX: event.clientX - 2,
      mouseY: event.clientY - 4,
    });
  };
  
  const handleContextMenuClose = () => {
    setContextMenu(null);
    setRightClickedLayerId(null);
  };
  
  const handleOption1Click = () => {
    if (rightClickedLayerId) {
      setMaskLayer(layers.find(l => l.id === rightClickedLayerId));
    }
    handleContextMenuClose();
  };
  
  const handleOption2Click = () => {
    if (rightClickedLayerId) {
      setProtectionLayer(layers.find(l => l.id === rightClickedLayerId));
    }
    handleContextMenuClose();
  };

   const handleSourceClick = () => {
    if (rightClickedLayerId) {
      setSourceLayer(layers.find(l => l.id === rightClickedLayerId));
    }
    handleContextMenuClose();
  };

  const handleTargetClick = () => {
    if (rightClickedLayerId) {
      setTargetLayer(layers.find(l => l.id === rightClickedLayerId));
    }
    handleContextMenuClose();
  };

  return (
    <Box
      sx={{
        width: '250px',
        bgcolor: '#333',
        borderLeft: '1px solid #424242',
        display: 'flex',
        flexDirection: 'column'
      }}
    >
      <Box
        sx={{
          padding: '0.5rem',
          borderBottom: '1px solid #424242',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center'
        }}
      >
        <Typography variant="subtitle2" sx={{ color: 'white' }}>Layers</Typography>
        <IconButton size="small" onClick={onAddLayer} sx={{ color: 'white' }}>
          <AddIcon fontSize="small" />
        </IconButton>
      </Box>
      <List sx={{ overflow: 'auto', flex: 1 }}>
        {layers.map((layer, index) => (
          <ListItem
          key={layer.id}
          onContextMenu={(e) => handleContextMenu(e, layer.id)}
          sx={{
            borderBottom: '1px solid #424242',
            cursor: 'pointer',
            paddingLeft: layer.type === 'mask' ? '24px' : '16px',
            bgcolor: (() => {

              if (layer.id === sourceLayer?.id) {
                return 'rgba(255, 165, 0, 0.1)'; // Orange for source
              }
              if (layer.id === targetLayer?.id) {
                return 'rgba(255, 192, 203, 0.1)'; // Pink for target
              }

              // Protection layer has highest priority
              if (layer.id === protectionLayer?.id) {
                return 'rgba(0, 255, 0, 0.1)'; // Green for protection
              }
              
              // Mask layer has second priority
              if (layer.id === maskLayer?.id) {
                return 'rgba(255, 0, 0, 0.1)'; // Red for mask
              }
          
              // Mask type styling
              if (layer.type === 'mask') {
                return selectedLayerId === layer.id
                  ? 'rgba(64, 126, 255, 0.15)'
                  : 'rgba(64, 126, 255, 0.08)';
              }
          
              // Regular layer styling
              return selectedLayerId === layer.id ? '#444' : 'transparent';
            })(),
          
            '&:hover': {
              bgcolor: (() => {
              if (layer.id === sourceLayer?.id) {
                return 'rgba(255, 165, 0, 0.15)';
              }
              if (layer.id === targetLayer?.id) {
                return 'rgba(255, 192, 203, 0.15)';
              }

                // Protection layer hover
                if (layer.id === protectionLayer?.id) {
                  return 'rgba(0, 255, 0, 0.15)';
                }
          
                // Mask layer hover
                if (layer.id === maskLayer?.id) {
                  return 'rgba(255, 0, 0, 0.15)';
                }
          
                // Mask type hover
                if (layer.type === 'mask') {
                  return selectedLayerId === layer.id
                    ? 'rgba(64, 126, 255, 0.2)'
                    : 'rgba(64, 126, 255, 0.12)';
                }
          
                // Regular layer hover
                return '#3a3a3a';
              })()
            }
          }}
          onClick={() => onSelectLayer(layer.id)}
        >
            <ListItemIcon sx={{ minWidth: 36 }}>
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  onToggleVisibility(layer.id);
                }}
                sx={{ color: 'white' }}
              >
                {layer.visible ? <VisibilityIcon fontSize="small" /> : <VisibilityOffIcon fontSize="small" />}
              </IconButton>
            </ListItemIcon>
            <ListItemText
              primary={layer.type === 'mask' ? getMaskName(layer) : `Layer ${index + 1}`}
              sx={{ 
                '& .MuiListItemText-primary': { 
                  color: 'white', 
                  fontSize: '0.875rem',
                  fontStyle: layer.type === 'mask' ? 'italic' : 'normal'
                } 
              }}
            />
            {layer.type === 'mask' && (
              <IconButton
                size="small"
                onClick={(e) => {
                  e.stopPropagation();
                  onApplyMask(layer.id);
                }}
                sx={{ color: 'white', mr: 1 }}
              >
                <ContentPasteOffIcon fontSize="small" />
              </IconButton>
            )}
            <IconButton
              size="small"
              onClick={(e) => {
                e.stopPropagation();
                onDeleteLayer(layer.id);
              }}
              sx={{ color: 'white' }}
            >
              <DeleteIcon fontSize="small" />
            </IconButton>
          </ListItem>
        ))}
                <Menu
          open={contextMenu !== null}
          onClose={handleContextMenuClose}
          anchorReference="anchorPosition"
          anchorPosition={
            contextMenu !== null
              ? { top: contextMenu.mouseY, left: contextMenu.mouseX }
              : undefined
          }
          sx={{
            '& .MuiPaper-root': {
              backgroundColor: '#1e1e1e', // Dark background
              border: '1px solid #333',
              boxShadow: '0 2px 10px rgba(0,0,0,0.3)',
            },
            '& .MuiMenuItem-root': {
              color: '#d4d4d4', // Light gray text
              '&:hover': {
                backgroundColor: '#2d2d2d', // Slightly lighter on hover
              }
            }
          }}
        >
          {rightClickedLayer?.type === 'mask' ? (
            <>
              <MenuItem 
                onClick={handleOption1Click}
                sx={{
                  '&:hover': { color: '#90caf9' } // Light blue hover
                }}
              >
                Select as Mask
              </MenuItem>
              <MenuItem 
                onClick={handleOption2Click}
                sx={{
                  '&:hover': { color: '#90caf9' }
                }}
              >
                Select as Protection
              </MenuItem>
            </>
          ) : (
            <>
            <MenuItem 
                onClick={handleMasksMerge}
                sx={{ 
                  color: '#ffa726', // Softer orange
                  '&:hover': {
                    backgroundColor: 'rgba(255, 167, 38, 0.08)',
                  }
                }}
              >
                Merge masks
              </MenuItem>
              <MenuItem 
                onClick={handleSourceClick}
                sx={{ 
                  color: '#ffa726', // Softer orange
                  '&:hover': {
                    backgroundColor: 'rgba(255, 167, 38, 0.08)',
                  }
                }}
              >
                Set as Source
              </MenuItem>
              <MenuItem 
                onClick={handleTargetClick}
                sx={{ 
                  color: '#f48fb1', // Softer pink
                  '&:hover': {
                    backgroundColor: 'rgba(244, 143, 177, 0.08)',
                  }
                }}
              >
                Set as Target
              </MenuItem>
            </>
          )}
        </Menu>
      </List>
    </Box>
  );
});

export default LayersPanel;