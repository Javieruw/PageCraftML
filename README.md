# PageCraftML - Graph Neural Network for Responsive Design

This module implements a Graph Neural Network (GNN) for predicting optimal responsive resize behavior from desktop HTML/CSS layouts to mobile sizes.

## Overview

The GNN analyzes the hierarchical structure of layout items and predicts:
- **Optimal breakpoints** for responsive design
- **Fluid scaling factors** for proportional resizing
- **Layout adaptations** (stacking, repositioning, hiding elements)
- **Media query values** for responsive CSS

## Architecture

### Graph Construction (`graph_utils.py`)
- Converts the Item hierarchy into a graph representation
- Each item becomes a node with features including:
  - Position and size (normalized)
  - Styling properties (colors, borders, shadows)
  - Layout properties (padding, margins)
  - Visual effects (filters, opacity)
- Edges represent parent-child relationships
- Edge features capture spatial relationships between parent and child elements

### GNN Model (`gnn_model.py`)
- **ResponsiveGNN**: Multi-layer Graph Convolutional Network with attention
- Uses GCN layers for feature propagation
- GAT (Graph Attention Network) layer for better feature learning
- Multiple prediction heads:
  - Scale predictor: predicts scaling factors (scale_x, scale_y)
  - Layout predictor: predicts new positions and sizes
  - Breakpoint predictor: predicts optimal breakpoint widths
  - Adaptation predictor: predicts layout adaptation type (scale/stack/hide/reposition)

### Responsive Predictor (`responsive_predictor.py`)
- Applies GNN predictions to actual Item objects
- Implements four adaptation strategies:
  1. **Scale**: Proportional resizing maintaining aspect ratios
  2. **Stack**: Vertical stacking for mobile layouts
  3. **Hide**: Hide elements that don't fit on mobile
  4. **Reposition**: Move elements to optimal positions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the server:
```bash
python nn_server.py
```

The server exposes a `/process` endpoint that accepts a `SavedWork` payload and returns optimized responsive layouts.

## Model Training

The current implementation uses randomly initialized weights. For production use:

1. Collect a dataset of desktop-to-mobile layout conversions
2. Train the model using supervised learning
3. Save the trained weights as `model_weights.pth`
4. The server will automatically load pretrained weights if available

## API

### POST `/process`

**Request:**
```json
{
  "payload": {
    "version": 1,
    "savedAt": "2024-01-01T00:00:00Z",
    "itemsByResolution": {
      "Desktop (1920x1080)": [...]
    },
    "gallery": []
  }
}
```

**Response:**
```json
{
  "processedPayload": {
    "version": 1,
    "savedAt": "2024-01-01T00:00:00Z",
    "itemsByResolution": {
      "Desktop (1920x1080)": [...],
      "Mobile (375x667)": [...]  // GNN-optimized mobile layout
    },
    "gallery": []
  }
}
```

## Files

- `nn_server.py`: FastAPI server with GNN integration
- `graph_utils.py`: Graph construction from Item hierarchy
- `gnn_model.py`: GNN model architecture
- `responsive_predictor.py`: Prediction application logic
- `requirements.txt`: Python dependencies

## Future Improvements

- [ ] Add training script with dataset
- [ ] Implement transfer learning from pretrained models
- [ ] Add support for tablet breakpoints
- [ ] Optimize for real-time inference
- [ ] Add confidence scores for predictions
- [ ] Support for custom breakpoint targets

