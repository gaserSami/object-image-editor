import React, { memo, useCallback } from 'react';
import { Switch, Box, Slider, Typography, Button, IconButton, TextField, Select, MenuItem } from '@mui/material';
import MoveIcon from '@mui/icons-material/OpenWith';
import SelectIcon from '@mui/icons-material/SelectAll';
import DeleteIcon from '@mui/icons-material/Delete';
import ContentCutIcon from '@mui/icons-material/ContentCut';
import AddIcon from '@mui/icons-material/Add';
import FileCopyIcon from '@mui/icons-material/FileCopy';
import ZoomOutMapIcon from '@mui/icons-material/ZoomOutMap';
import ColorLensIcon from '@mui/icons-material/ColorLens';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import HealingIcon from '@mui/icons-material/Healing';
import NearMe from '@mui/icons-material/NearMe';
import RoundedCornerIcon from '@mui/icons-material/RoundedCorner';

const OptionsBar = memo(function OptionsBar({ selectedTool, onCreateMask,
  onRectTool, onSelect,
  onResetIndicators, brushSize,
  setBrushSize, iterations, setIterations,
  onRemove, onRetargetApply,
  retargetWidth, onRetargetWidthChange,
  retargetHeight, onRetargetHeightChange,
  onBlend,
  blendMode,
  setBlendMode,
  isForward,
  setIsForward,
  onAddPathOffset,
  onHeal,
  onHealAI,
  layers,
  referenceLayerId,
  onReferenceLayerChange,
  retargetMode,
  onRetargetModeChange,
  direction,
  setDirection,
  considerScale,
  setConsiderScale
}) {
  const getToolIcon = useCallback(() => {
    switch (selectedTool) {
      case 'pointer': return <NearMe fontSize="small" />;
      case "lasso": return <RoundedCornerIcon fontSize="small" />;
      case 'select': return <SelectIcon fontSize="small" />;
      case 'remove': return <DeleteIcon fontSize="small" />;
      case 'add': return <AddIcon fontSize="small" />;
      case 'retarget': return <AutoFixHighIcon fontSize="small" />;
      case 'heal': return <HealingIcon fontSize="small" />;
      case 'cut': return <ContentCutIcon fontSize="small" />;
      case 'move': return <MoveIcon fontSize="small" />;
      case 'duplicate': return <FileCopyIcon fontSize="small" />;
      case 'scale': return <ZoomOutMapIcon fontSize="small" />;
      case 'color': return <ColorLensIcon fontSize="small" />;
      default: return null;
    }
  }, [selectedTool]);

  const handleBrushSizeChange = (event, newValue) => {
    setBrushSize(newValue);
  };

  const renderToolOptions = () => {
    switch (selectedTool) {
      case 'pointer':
        return (
          <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
            Click and drag to move the image.
          </Typography>
        );
      case "lasso":
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {"Select layer then draw selection freely."}
            </Typography>
            <Button
              onClick={() => onCreateMask()}  // Ensure this calls the handler
              sx={{
                color: 'white',
                fontSize: '0.5rem'
              }}
              size="small"
              variant="contained"
            >
              Create Mask
            </Button>
          </Box>
        );
      case "select":
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {"Select layer then start selection."}
            </Typography>

            <Button
              onClick={() => onRectTool()}
              sx={{
                color: 'white',
                fontSize: '0.5rem'
              }}
              size="small"
              variant="contained"
            >
              Intial Selection
            </Button>

            <Box display="flex" alignItems="center" width={150}>
              <Typography variant="caption" sx={{ fontSize: '0.75rem', minWidth: 80, paddingRight: "5px" }}>Brush Size</Typography>
              <Slider
                value={typeof brushSize === 'number' ? brushSize : 0}
                onChange={handleBrushSizeChange}
                min={1}
                max={100}
                step={1}
                valueLabelDisplay="auto"
                size="small"
                sx={{
                  width: '100%',
                  '& .MuiSlider-thumb': {
                    transition: 'transform 0.2s'
                  }
                }}
              />
            </Box>

            <Box display="flex" alignItems="center" width={150}>
              <Typography variant="caption" sx={{ color: 'white', minWidth: 60 }}>
                Iterations
              </Typography>
              <Slider
                value={iterations}
                onChange={(e, newValue) => setIterations(Number(newValue))}
                valueLabelDisplay="auto"
                step={1}
                min={1}
                max={100}
                size="small"
                sx={{
                  width: '100%',
                  '& .MuiSlider-thumb': {
                    transition: 'transform 0.2s'
                  }
                }}
              />
            </Box>

            <Button
              onClick={() => onSelect()}
              sx={{
                color: 'white',
                fontSize: '0.5rem'
              }}
              size="small"
              variant="contained"
            >
              Enhance Selection
            </Button>

            <Button
              onClick={() => onAddPathOffset()}
              sx={{
                color: 'white',
                fontSize: '0.5rem'
              }}
              size="small"
              variant="contained"
            >
              Add Offset
            </Button>

            <Button
              onClick={() => onResetIndicators()}
              sx={{
                color: 'white',
                fontSize: '0.5rem'
              }}
              size="small"
              variant="contained"
            >
              Reset Indictors
            </Button>

            <Button
              onClick={() => onCreateMask()}
              sx={{
                color: 'white',
                fontSize: '0.5rem'
              }}
              size="small"
              variant="contained"
            >
              Create Mask
            </Button>

          </Box>
        );
      case 'remove':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              Select your target layer, protection layer and mask layer then click to remove.
            </Typography>
            <Select
              size="small"
              value={direction}
              onChange={(e) => setDirection(e.target.value)}
              sx={{ 
                width: '120px',
                color: 'white',
                height: '24px',
                fontSize: '0.75rem',
                '& .MuiOutlinedInput-notchedOutline': {
                  borderColor: '#90caf9',
                },
                '& .MuiSelect-select': {
                  padding: '2px 8px',
                  lineHeight: '20px',
                }
              }}
            >
              <MenuItem value="horizontal" sx={{ fontSize: '0.75rem' }}>Horizontal</MenuItem>
              <MenuItem value="vertical" sx={{ fontSize: '0.75rem' }}>Vertical</MenuItem>
              <MenuItem value="auto" sx={{ fontSize: '0.75rem' }}>Auto</MenuItem>
            </Select>
            <Switch
              checked={isForward}
              onChange={(e) => setIsForward(e.target.checked)}
              size="small"
              color="primary"
            />
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {isForward ? 'Forward' : 'Backward'} Energy
            </Typography>
            <Button
              variant="contained" color="primary" size="small" sx={{ fontSize: '0.5rem' }}
              onClick={() => onRemove()}
            >
              Remove
            </Button>
          </Box>
        );
      case 'add':
        const blendModes = ['Max', 'Average', 'Replace', 'Sum'];

        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              Select your source layer, target layer and mask layer then click to blend.
            </Typography>
            <Select
              value={blendMode}
              onChange={(e) => setBlendMode(e.target.value)}
              size="small"
              sx={{ minWidth: 120, fontSize: '0.75rem' }}
            >
              {blendModes.map((mode) => (
                <MenuItem key={mode} value={mode}>
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </MenuItem>
              ))}
            </Select>
            <Button
              variant="contained"
              color="primary"
              size="small"
              sx={{ fontSize: '0.5rem' }}
              onClick={() => onBlend(blendMode)}
            >
              Blend
            </Button>
          </Box>
        );
      case 'move':
        return (
          <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
            Drag selected objects to move them.
          </Typography>
        );
      case 'duplicate':
        return (
          <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
            Select and drag to duplicate objects.
          </Typography>
        );
      case 'scale':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>Scale:</Typography>
            <Slider
              defaultValue={100}
              min={10}
              max={200}
              size="small"
              sx={{
                width: 60,
                color: '#90caf9',
                padding: '5px 0' // Optional: reduces vertical padding
              }}
              aria-label="scale"
            />
          </Box>
        );
      case 'transform':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>Rotation:</Typography>
            <Slider
              defaultValue={0}
              min={-180}
              max={180}
              sx={{ width: 80, color: '#90caf9' }}
              aria-label="rotation"
            />
          </Box>
        );
      case 'color':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>Color:</Typography>
            <input type="color" style={{ width: '24px', height: '24px' }} />
          </Box>
        );
      case 'retarget':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              Adjust target image and optionally select protection layer:
            </Typography>
            <Box display="flex" alignItems="center" gap={1}>
              <Select
                size="small"
                value={retargetMode}
                onChange={(e) => onRetargetModeChange(e.target.value)}
                sx={{ 
                  width: '120px',
                  color: 'white',
                  height: '24px',
                  fontSize: '0.75rem',
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: '#90caf9',
                  },
                  '& .MuiSelect-select': {
                    padding: '2px 8px',
                    lineHeight: '20px',
                  }
                }}
              >
                <MenuItem value="percentage" sx={{ fontSize: '0.75rem' }}>Percentage</MenuItem>
                <MenuItem value="reference" sx={{ fontSize: '0.75rem' }}>Reference Layer</MenuItem>
              </Select>

              {retargetMode === 'percentage' ? (
                <>
                  <Typography variant="caption">Width %:</Typography>
                  <TextField
                    type="number"
                    size="small"
                    value={retargetWidth}
                    onChange={(e) => { onRetargetWidthChange(e.target.value) }}
                    InputProps={{
                      inputProps: { min: -80, max: 80 },
                      sx: {
                        color: 'white',
                        height: '32px',
                        '& input': {
                          padding: '4px 8px',
                          fontSize: '0.75rem',
                        },
                        '& .MuiOutlinedInput-notchedOutline': {
                          borderColor: '#90caf9',
                        },
                        backgroundColor: 'transparent',
                      }
                    }}
                    sx={{ width: '80px' }}
                  />

                  <Typography variant="caption" sx={{ color: 'white' }}>Height %:</Typography>
                  <TextField
                    type="number"
                    size="small"
                    value={retargetHeight}
                    onChange={(e) => { onRetargetHeightChange(e.target.value) }}
                    InputProps={{
                      inputProps: { min: -100, max: 100 },
                      sx: {
                        color: 'white',
                        height: '32px',
                        '& input': {
                          padding: '4px 8px',
                          fontSize: '0.75rem',
                        },
                        '& .MuiOutlinedInput-notchedOutline': {
                          borderColor: '#90caf9',
                        },
                        backgroundColor: 'transparent',
                      }
                    }}
                    sx={{ width: '80px' }}
                  />
                </>
              ) : (
                <Select
                  size="small"
                  value={referenceLayerId || ''}
                  onChange={(e) => onReferenceLayerChange(e.target.value)}
                  sx={{ 
                    width: '150px',
                    color: 'white',
                    height: '24px',
                    fontSize: '0.75rem',
                    '& .MuiOutlinedInput-notchedOutline': {
                      borderColor: '#90caf9',
                    },
                    '& .MuiSelect-select': {
                      padding: '2px 8px',
                      lineHeight: '20px',
                    }
                  }}
                >
                  {layers.map((layer) => (
                    <MenuItem 
                      key={layer.id} 
                      value={layer.id}
                      sx={{ fontSize: '0.75rem' }}
                    >
                      {layer.type === 'mask' ? 
                        `Mask (${layer.parentLayerId ? `Layer ${layers.findIndex(l => l.id === layer.parentLayerId) + 1}` : 'Unlinked'})` : 
                        `Layer ${layers.indexOf(layer) + 1}`}
                    </MenuItem>
                  ))}
                </Select>
              )}
              <Switch
                checked={considerScale}
                onChange={(e) => setConsiderScale(e.target.checked)}
                size="small"
                color="primary"
              />
              <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                Consider Scale
              </Typography>
            </Box>
            <Switch
              checked={isForward}
              onChange={(e) => setIsForward(e.target.checked)}
              size="small"
              color="primary"
            />
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
              {isForward ? 'Forward' : 'Backward'} Energy
            </Typography>
            <Button
              variant="contained"
              color="primary"
              size="small"
              onClick={onRetargetApply}
              sx={{ fontSize: '0.75rem' }}
            >
              Apply
            </Button>
          </Box>
        );
          case 'heal':
          return (
            <Box display="flex" flexDirection="row" gap={1} p={1} justifyContent={'center'} alignItems={'center'}>
              <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>
                Select target layer and mask layer then click to heal.
              </Typography>
              <Button
                variant="contained"
                color="primary"
                size="small"
                onClick={onHeal}
                sx={{
                  color: 'white',
                  fontSize: '0.5rem'
                }}
              >
                Apply Healing
              </Button>
              <Button
                variant="contained"
                size="small"
                onClick={() => {onHealAI()}}
                startIcon={<AutoFixHighIcon />}
                sx={{
                  fontSize: '0.5rem',
                  background: 'linear-gradient(45deg, #FF8E53 30%, #FE6B8B 90%)',
                  border: 0,
                  boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
                  color: 'white',
                  '&:hover': {
                    background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
                    boxShadow: '0 3px 7px 2px rgba(255, 105, 135, .5)',
                  },
                  animation: 'glowing 1.5s infinite',
                  '@keyframes glowing': {
                    '0%': { boxShadow: '0 0 5px #FF8E53' },
                    '50%': { boxShadow: '0 0 20px #FE6B8B' },
                    '100%': { boxShadow: '0 0 5px #FF8E53' }
                  }
                }}
              >
                Heal with AI
              </Button>
            </Box>
          );
      case 'brush':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.5rem' }}>Brush Size:</Typography>
            <Slider

              defaultValue={5}
              min={1}
              max={20}
              size="small"
              sx={{
                width: 40,
                color: '#90caf9',
                padding: '5px 0'
              }}
              aria-label="brush size"
            />
          </Box>
        );
      case 'crop':
        return (
          <Box display="flex" alignItems="center" gap={1}>
            <Typography variant="body2" sx={{ fontSize: '0.75rem' }}>Crop Tool:</Typography>
            <Button variant="contained" color="primary" size="small" sx={{ fontSize: '0.5rem' }}>
              Apply Crop
            </Button>
          </Box>
        );
      default:
        return null;
    }
  };

  return (
    <Box
      sx={{
        bgcolor: '#444',
        color: '#fff',
        padding: '0.25rem 0.5rem',
        borderBottom: '1px solid #424242',
        display: 'flex',
        alignItems: 'center',
        height: '40px',
      }}
    >
      <IconButton sx={{ color: 'white', fontSize: '1rem' }} >
        {getToolIcon()}
      </IconButton>
      {renderToolOptions()}
    </Box>
  );
});

export default OptionsBar;