import React, { memo, useCallback } from 'react';
import Box from '@mui/material/Box';
import Toolbar from '@mui/material/Toolbar';
import Button from '@mui/material/Button';
import Menu from '@mui/material/Menu';
import MenuItem from '@mui/material/MenuItem';

const MENU_ITEMS = {
  file: ['New', 'Open', 'Save', 'Export as PNG', 'Export as JPG'],
  edit: ['Undo', 'Redo'],
  image: ['Adjustments', 'Resize'],
  layer: ['New Layer', 'Delete Layer'],
  select: ['All', 'Deselect'],
  filters: ['Brightness', 'Contrast', 'Saturation'],
  view: ['Zoom In', 'Zoom Out']
};

const exportImage = (layers, selectedLayerId, format) => {
  if (!selectedLayerId || !layers.length) return;
  const selectedLayer = layers.find(layer => layer.id === selectedLayerId);
  if (!selectedLayer) return;
  const tempCanvas = document.createElement('canvas');
  const tempCtx = tempCanvas.getContext('2d');
  const image = selectedLayer.image;
  tempCanvas.width = image.width;
  tempCanvas.height = image.height;
  tempCtx.drawImage(image, 0, 0);
  const link = document.createElement('a');
  link.download = `export-${Date.now()}.${format.toLowerCase()}`;
  link.href = tempCanvas.toDataURL(`image/${format.toLowerCase()}`);
  link.click();
};

const MenuButton = memo(({ label, onClick }) => (
  <Button
    onClick={onClick}
    sx={{
      color: 'white',
      textTransform: 'none',
      fontSize: '0.75rem',
      minWidth: 'auto',
      padding: '0.1rem 0.5rem'
    }}
  >
    {label}
  </Button>
));

const CustomMenu = memo(({ items, anchorEl, onClose, onExport }) => (
  <Menu
    anchorEl={anchorEl}
    open={Boolean(anchorEl)}
    onClose={onClose}
    sx={{ '& .MuiPaper-root': { bgcolor: '#2b2b2b', color: '#fff' } }}
    anchorOrigin={{ vertical: 'bottom', horizontal: 'left' }}
    transformOrigin={{ vertical: 'top', horizontal: 'left' }}
  >
    {items.map((item) => (
      <MenuItem
        key={item}
        onClick={() => {
          if (item === 'Export as PNG') {
            onExport('PNG');
          } else if (item === 'Export as JPG') {
            onExport('JPEG');
          }
          onClose();
        }}
        sx={{ fontSize: '0.75rem' }}
      >
        {item}
      </MenuItem>
    ))}
  </Menu>
));

const ImageEditorMenuBar = memo(({ layers, selectedLayerId }) => {
  const [menuState, setMenuState] = React.useState({
    activeMenu: null,
    anchorEl: null
  });

  const handleMenuClick = useCallback((menuName) => (event) => {
    setMenuState({
      activeMenu: menuName,
      anchorEl: event.currentTarget
    });
  }, []);

  const handleMenuClose = useCallback(() => {
    setMenuState({
      activeMenu: null,
      anchorEl: null
    });
  }, []);

  const handleExport = useCallback((format) => {
    exportImage(layers, selectedLayerId, format);
  }, [layers, selectedLayerId]);

  return (
    <Box sx={{ flexGrow: 1, bgcolor: '#2b2b2b', padding: "0" }}>
      <Toolbar
        variant="dense"
        sx={{
          bgcolor: '#333',
          color: '#fff',
          display: 'flex',
          minHeight: '48px',
          padding: '12px !important'
        }}
      >
        {Object.keys(MENU_ITEMS).map((menuName) => (
          <MenuButton
            key={menuName}
            label={menuName.charAt(0).toUpperCase() + menuName.slice(1)}
            onClick={handleMenuClick(menuName)}
          />
        ))}
      </Toolbar>
      <CustomMenu
        items={menuState.activeMenu ? MENU_ITEMS[menuState.activeMenu] : []}
        anchorEl={menuState.anchorEl}
        onClose={handleMenuClose}
        onExport={handleExport}
      />
    </Box>
  );
});

export default ImageEditorMenuBar;