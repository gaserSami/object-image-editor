import React, {memo, useCallback} from 'react';
import { Toolbar, IconButton, Tooltip } from '@mui/material';
import SelectIcon from '@mui/icons-material/SelectAll';
import DeleteIcon from '@mui/icons-material/Delete';
import AddIcon from '@mui/icons-material/Add';
import MoveIcon from '@mui/icons-material/OpenWith';
import FileCopyIcon from '@mui/icons-material/FileCopy';
import ZoomOutMapIcon from '@mui/icons-material/ZoomOutMap';
import ColorLensIcon from '@mui/icons-material/ColorLens';
import AutoFixHighIcon from '@mui/icons-material/AutoFixHigh';
import HealingIcon from '@mui/icons-material/Healing';
import NearMeIcon from '@mui/icons-material/NearMe';
import RoundedCornerIcon from '@mui/icons-material/RoundedCorner';
import CropIcon from '@mui/icons-material/Crop';
import RotateRightIcon from '@mui/icons-material/RotateRight';
import TextFieldsIcon from '@mui/icons-material/TextFields';


const CustomToolbar = memo(function CustomToolbar({ onSelectTool, selectedTool }) {
  const tools = [
    { id: 'pointer', icon: NearMeIcon, tooltip: 'Pointer' },
    { id: 'lasso', icon: RoundedCornerIcon, tooltip: 'Lasso' },
    { id: 'select', icon: SelectIcon, tooltip: 'Object Selection' },
    { id: 'remove', icon: DeleteIcon, tooltip: 'Remove Object' },
    { id: 'add', icon: AddIcon, tooltip: 'Add Object' },
    { id: 'move', icon: MoveIcon, tooltip: 'Move' },
    { id: 'duplicate', icon: FileCopyIcon, tooltip: 'Duplicate' },
    { id: 'scale', icon: ZoomOutMapIcon, tooltip: 'Scale' },
    { id: 'color', icon: ColorLensIcon, tooltip: 'Change Color' },
    { id: 'retarget', icon: AutoFixHighIcon, tooltip: 'Retarget' },
    { id: 'heal', icon: HealingIcon, tooltip: 'Healing' },
    { id: 'crop', icon: CropIcon, tooltip: 'Crop' },
    { id: 'rotate', icon: RotateRightIcon, tooltip: 'Rotate' },
    { id: 'text', icon: TextFieldsIcon, tooltip: 'Text' },
  ];

  const buttonStyle = useCallback(
    (tool) => ({
      color: selectedTool === tool ? '#90caf9' : 'white',
      '&:hover': {
        backgroundColor: 'rgba(144, 202, 249, 0.08)'
      }
    }), [selectedTool]
  );

  const handleToolClick = useCallback((toolId) => {
    onSelectTool(toolId);
  }, [onSelectTool]);

  return (
    <Toolbar
      sx={{
        bgcolor: '#333',
        flexDirection: 'column',
        paddingLeft: '0 !important',
        paddingRight: '0 !important',
        gap: '2px',
        width: '51px',
        borderRight: '1px solid #424242',
      }}
    >
      {tools.map((tool) => (
        <React.Fragment key={tool.id}>
          <Tooltip title={tool.tooltip} placement="right">
            <IconButton
              onClick={() => handleToolClick(tool.id)}
              sx={buttonStyle(tool.id)}
            >
              <tool.icon fontSize="small" />
            </IconButton>
          </Tooltip>
        </React.Fragment>
      ))}
    </Toolbar>
  );
});

export default CustomToolbar;