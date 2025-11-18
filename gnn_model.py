"""
Graph Neural Network model for predicting responsive resize behavior.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_max_pool
from torch_geometric.data import Data
from typing import Dict, Tuple


class ResponsiveGNN(nn.Module):
    """
    Graph Neural Network for predicting optimal responsive resize behavior.
    
    Predicts:
    - Optimal breakpoints
    - Fluid scaling factors
    - Layout adaptations (stacking, hiding, repositioning)
    - Media query values
    """
    
    def __init__(
        self,
        input_dim: int = 20,
        hidden_dim: int = 128,
        output_dim: int = 8,  # [scale_x, scale_y, new_x, new_y, new_width, new_height, breakpoint, layout_adaptation]
        num_layers: int = 3,
        dropout: float = 0.2
    ):
        super(ResponsiveGNN, self).__init__()
        
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # Attention mechanism for better feature learning
        self.attention = GATConv(hidden_dim, hidden_dim, heads=4, dropout=dropout, concat=False)
        
        # Batch normalization
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Output layers for different predictions
        # Scaling factors
        self.scale_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # scale_x, scale_y
        )
        
        # New position and size
        self.layout_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4)  # new_x, new_y, new_width, new_height
        )
        
        # Breakpoint prediction (when to apply changes)
        self.breakpoint_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # breakpoint width
            nn.Sigmoid()  # Normalize to 0-1
        )
        
        # Layout adaptation type (0=scale, 1=stack, 2=hide, 3=reposition)
        self.adaptation_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 4),  # probabilities for each adaptation type
            nn.Softmax(dim=-1)
        )
    
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the GNN.
        
        Returns:
            Dictionary with predictions for each node:
            - scales: (N, 2) tensor of [scale_x, scale_y]
            - layouts: (N, 4) tensor of [x, y, width, height]
            - breakpoints: (N, 1) tensor of breakpoint widths
            - adaptations: (N, 4) tensor of adaptation type probabilities
        """
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        # Graph convolutions
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            x = self.batch_norms[i](x)
            x = F.relu(x)
            x = self.dropout(x)
        
        # Attention layer
        x = self.attention(x, edge_index)
        x = F.relu(x)
        
        # Global pooling for graph-level features
        if batch is not None:
            graph_features = torch.cat([
                global_mean_pool(x, batch),
                global_max_pool(x, batch)
            ], dim=1)
        else:
            # Single graph
            graph_features = torch.cat([
                x.mean(dim=0, keepdim=True),
                x.max(dim=0, keepdim=True)[0]
            ], dim=1)
        
        # Expand graph features to match node count
        if batch is not None:
            graph_features_expanded = graph_features[batch]
        else:
            graph_features_expanded = graph_features.expand(x.size(0), -1)
        
        # Combine node and graph features
        combined_features = torch.cat([x, graph_features_expanded], dim=1)
        
        # Predictions
        scales = self.scale_predictor(combined_features)
        layouts = self.layout_predictor(combined_features)
        breakpoints = self.breakpoint_predictor(combined_features)
        adaptations = self.adaptation_predictor(combined_features)
        
        return {
            'scales': scales,
            'layouts': layouts,
            'breakpoints': breakpoints,
            'adaptations': adaptations
        }


def predict_responsive_behavior(
    model: ResponsiveGNN,
    graph: Data,
    target_width: float = 375.0,  # Mobile width
    target_height: float = 667.0,  # Mobile height
    source_width: float = 1920.0,  # Desktop width
    source_height: float = 1080.0  # Desktop height
) -> Dict:
    """
    Use the GNN model to predict responsive behavior for a graph.
    
    Returns predictions for each node in the graph.
    """
    model.eval()
    device = next(model.parameters()).device
    graph = graph.to(device)
    
    with torch.no_grad():
        predictions = model(graph)
    
    # Convert predictions to CPU numpy arrays
    scales = predictions['scales'].cpu().numpy()
    layouts = predictions['layouts'].cpu().numpy()
    breakpoints = predictions['breakpoints'].cpu().numpy()
    adaptations = predictions['adaptations'].cpu().numpy()
    
    # Scale breakpoints to actual pixel values
    breakpoints_scaled = breakpoints * source_width  # Convert normalized to pixels
    
    # Get adaptation type (argmax)
    adaptation_types = adaptations.argmax(axis=1)
    
    return {
        'scales': scales,
        'layouts': layouts,
        'breakpoints': breakpoints_scaled,
        'adaptations': adaptation_types,
        'adaptation_probs': adaptations
    }

