# Object-Image-Editor

This project is an Object Image Editor Which focuses on object editing.

## Prerequisites

Before running the project, ensure you have the following installed:

- **Python** (version 3.8+)
- **Node.js** (version 14+)
- **npm** (Node Package Manager)
- **virtualenv** (for creating a Python virtual environment)

---

## Steps to Run the Project

### 1. Backend Setup

1. **Navigate to the backend folder**
   ```bash
   cd backend
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   ```

3. **Activate the virtual environment**
   - **For Windows**:
     ```bash
     venv\Scripts\activate
     ```
   - **For macOS/Linux**:
     ```bash
     source venv/bin/activate
     ```

4. **Install the required Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Download the LAMA Model**
   - Download the model file from [this URL](https://github.com/Sanster/models/releases/download/add_big_lama/big-lama.pt).
   - Save the downloaded `big-lama.pt` file in the following location:
     ```bash
     backend/assets/big-lama.pt
     ```

6. **Run the backend server**
   ```bash
   python run.py
   ```

   This starts the backend server on the default port (e.g., `http://127.0.0.1:5000`).

---

### 2. Frontend Setup

1. **Navigate to the frontend folder**
   ```bash
   cd ../frontend
   ```

2. **Install frontend dependencies**
   ```bash
   npm install
   ```

3. **Start the frontend development server**
   ```bash
   npm start
   ```

   This starts the frontend server on the default port (e.g., `http://localhost:3000`).

---

## Detailed Usage Guide

### Adding Images and Masks
1. **Adding New Layers**:
   - Click the plus icon (+) in the layers section
   - You will be prompted with two options:
     - Add an image
     - Add a predefined mask
   - **Important Note for Masks**: 
     - Masks must be associated with a specific image layer
     - Before adding a mask, ensure you have at least one image layer
     - When adding a mask, you'll need to select which image layer it will be associated with

### Tools and Features

#### Pointer Tool (First Tool)
1. **Activation and Basic Movement**:
   - Click the pointer tool in the toolbar (first tool)
   - Select any layer by clicking on it in the layers section
   - Click and drag the selected layer to move it within the canvas

2. **Resizing Capabilities**:
   - When pointer tool is active, resizing handles appear on the image
   - Drag these handles to scale the image to desired size

3. **Rotation Features**:
   - Special rotate handles appear on the image
   - Drag these handles to rotate the image to any angle

#### Layers Management
1. **Layer Controls**:
   - All layers are displayed in the layers section
   - Click any layer to make it the active layer (appears light gray)
   - Non-active layers appear dark gray
   - Toggle visibility using the hide/unhide button (eye icon)
   - Delete any layer using the delete button
   - Masks appear indented under their associated layers

2. **Mask Operations**:
   - For mask layers, an "Apply Mask" button is available
   - Clicking this button creates a new layer
   - The new layer shows the result of applying the mask to its associated image
   - When a layer has multiple masks:
     - Option to create a merged mask becomes available
     - Merged mask combines all masks associated with that layer

3. **Layer Role Indicators**:
   - Source layer appears yellow-brown
   - Target layer appears light saturated brown
   - Mask selected as mask appears brick red
   - Mask selected as protection appears green

#### Interface Elements
1. **Toolbar Interaction**:
   - Click any toolbar item to view its specific options
   - All options appear in the options bar (located under the header)

2. **Header Menu**:
   - Contains main options: File, Edit, Image, Layer, Select, Filters, View
   - Currently implemented features:
     - File > Export: Exports the currently selected layer
     - Other menu items are present but functionality is pending

### Advanced Selection Tools

#### Lasso Tool
1. **Selection Process**:
   - Draw freeform selections directly on the canvas
   - Selections are independent of any specific image
   - Can be used with any image layer
   - Selection remains active until explicitly removed (press Escape key)

2. **Creating Masks**:
   - Click "Create Mask" in the options bar
   - Important requirements:
     - Must have a layer selected
     - Selected layer must be under the selection area
   - Mask creation considers:
     - The relative position between selection and image
     - Current position of the image under the selection
     - Transformation of the image in canvas space

3. **Workflow Flexibility**:
   - Move images freely under the selection
   - Apply the same selection to different images
   - Adjust image position before applying mask
   - Selection persists for multiple uses until removed
   - Created mask automatically associates with the currently selected layer

4. **Key Notes**:
   - Masks are created based on current spatial relationships
   - Selection-to-image relationship determines mask outcome
   - Selection can be reused multiple times with different layers
   - Press Escape key to remove selection when finished

#### Object Selection Tool
1. **Initial Selection Process**:
   - Click "Initial Selection" in the options bar
   - Enter drawing mode
   - Draw a rectangle around the target object
   - Like Lasso tool, selection is independent of layers
   - Selected layer must be under the selection area
   - Selection works based on relative positions in canvas space

2. **Selection Enhancement**:
   - Click "Enhance Selection" for initial automatic selection
   - Use indicator drawing mode for refinement:
     - Press "1" key: Draw green lines for foreground
     - Press "0" key: Draw red lines for background
     - Press "3" key: Exit drawing mode
   - Adjust brush size for more precise indicator drawing
   - Default iteration count is 5 (works well in most cases)
   - Can modify number of iterations if needed for better results

3. **Refinement Controls**:
   - "Reset Indicators" button: Clears all foreground/background lines
   - Repeat enhancement process as needed
   - Add offset before creating final mask for safe margins in object removal
   - Full flexibility to move images under selection before finalizing

### Advanced Tools

#### Object Removal Tool
1. **Layer Selection Process**:
   - Select the removal tool from the toolbar
   - Right-click on layers in the layers panel to designate their roles:
     - Select "Select as Source" for source layer
     - Select "Select as Target" for target layer
     - Select "Select as Mask" for mask layer
     - Select "Select as Protection" for protection layer (optional)

2. **Layer Roles Explanation**:
   - **Target Layer**: The image you want to remove an object from
   - **Mask Layer**: Defines the object to be removed
   - **Protection Layer**: (Optional) Specifies areas that should remain untouched during removal
   
3. **Removal Options**:
   - Choose between two energy calculation methods:
     - Forward Energy
     - Backward Energy
   - Click the removal button to initiate the process

4. **Important Notes**:
   - This is a destructive removal process
   - The resulting image will be smaller as pixels are actually deleted
   - Particularly useful when:
     - Exemplar-based inpainting fails
     - Image has busy or non-uniform patterns
   - The reduced size can be later adjusted using the retargeting tool

#### Seamless Adding Tool
1. **Layer Selection Process**:
   - Select the seamless adding tool from the toolbar
   - Right-click on layers in the layers panel to set their roles:
     - Select "Select as Source" for source layer
     - Select "Select as Target" for target layer
     - Select "Select as Mask" for mask layer

2. **Layer Roles Explanation**:
   - **Source Layer**: The image containing the object you want to add
   - **Target Layer**: The destination image where you want to add the object
   - **Mask Layer**: Defines which parts of the source image to use

3. **Blending Modes**:
   - Choose from several blending options:
     - **Average**: Averages the pixels from both source and target
     - **Max**: Uses the pixel with maximum effect from either source or target
     - **Replace**: Completely replaces target pixels with source pixels
     - **Sum**: Adds pixel values from both images (similar to average but brighter)

4. **Applying the Blend**:
   - Select your preferred blending mode
   - Click the blend button to apply the effect
   - Experiment with different modes to find the best result for your specific case

#### Seamless Adding Tool Detailed Workflow
1. **Preparing the Source Layer**:
   - Start with the layer containing the object you want to extract
   - Create a selection around the desired object
     - Selection should be larger than the actual object
     - Precision is not critical at this stage
     - Include some surrounding area for better results
   - Create and apply a mask to isolate this section
   - This creates a new layer with just the selected area

2. **Setting Up the Target Layer**:
   - Place your target layer in the canvas
   - Position it exactly where you want the final result
   - Apply any necessary transformations
     - Scale, rotate, or move as needed
     - These transformations will be considered in the final blend

3. **Creating the Working Mask**:
   - **Important**: Select the source layer before this step
   - Create a new selection around the actual part you want in the target
     - Selection doesn't need to be extremely precise
     - Should focus on the main object area
   - Create a mask from this selection
     - The mask will automatically align with the source layer
     - This alignment is crucial for proper blending

4. **Final Blending**:
   - Select the appropriate blending mode
     - Max mode typically produces the best results
   - Apply the seamless addition
   - Adjust if needed by trying different blending modes

#### Retargeting Tool
1. **Layer Selection Process**:
   - Select the retargeting tool from the toolbar
   - Right-click on layers in the layers panel to set roles:
     - Select "Select as Target" for the layer to be retargeted
     - Select "Select as Protection" for protection layer (optional)

2. **Size Adjustment Controls**:
   - Enter percentage values in width and height fields:
     - Positive values: Expand the image
     - Negative values: Reduce the image
   - Use keyboard keys to incrementally adjust percentages
   - Values represent the desired change relative to current size

3. **Energy Calculation Options**:
   - Choose between:
     - Forward Energy
     - Backward Energy

4. **Protection Features**:
   - Optional protection layer to preserve specific regions
   - Protected areas maintain their integrity during resizing
   - Particularly useful for preserving important image elements

5. **Applying Retargeting**:
   - Click apply after setting desired parameters
   - Process maintains important image details
   - Avoids traditional scaling artifacts like stretching or compression

#### Healing Tool
1. **Layer Selection Process**:
   - Select the healing tool from the toolbar
   - Right-click on layers in the layers panel to set roles:
     - Select "Select as Target" for the layer to be healed
     - Select "Select as Mask" for mask layer

2. **Healing Options**:
   - Two available healing methods:
     - **Standard Healing**: Uses traditional healing algorithms
     - **AI-Powered Healing**: Utilizes AI models for healing

3. **Application Process**:
   - Select your preferred healing method:
     - Click "Apply Healing" for standard healing
     - Click "Heal with AI" for AI-based healing
   - Results vary based on image content and scenario
   - Try both methods to determine the best outcome for your specific case

4. **Important Notes**:
   - Non-destructive removal alternative
   - Different from object removal as it preserves image dimensions
   - Each healing method may perform better in different scenarios
   - Experimentation recommended to find optimal results

### Important Notes About Canvas Transformations

1. **Temporary Nature**:
   - Canvas transformations (rotation, scaling, position) are temporary
   - When exporting an image, it's saved in its original orientation
   - Example: A rotated image will be downloaded in its original orientation

2. **Use Cases**:
   - **Comparison**: 
     - Useful for comparing results side by side
     - Example: Scale an image to compare with its retargeted version
   
   - **Operations**:
     - Some tools consider canvas transformations during processing
     - Example: Seamless addition tool takes source layer transformations into account
     - Allows flexibility in positioning and scaling objects before adding them

3. **Workflow Benefits**:
   - Provides visual flexibility during editing
   - Enables precise positioning for comparisons
   - Supports interactive adjustments without affecting final output

---

## Example Workflow
1. Run the backend and frontend servers as described earlier.
2. Use the **plus button** in the layers section to load an image.
3. Select a tool like **Lasso** or **Object Selection** to make a selection.
4. Use options in the **options bar** to refine or apply masks.
5. Perform operations like masking, removing, or blending.
6. Export the final result using **File > Export**.

---

## Summary of Commands

Here’s a quick reference:

```bash
# Backend
cd backend
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate    # For Windows
pip install -r requirements.txt
# Place big-lama.pt in backend/assets/
python run.py

# Frontend
cd ../frontend
npm install
npm start
```

---

### Notes
- Ensure the `big-lama.pt` model file is correctly placed in the `assets` folder under `backend`.
- The backend and frontend servers need to be running simultaneously.

---