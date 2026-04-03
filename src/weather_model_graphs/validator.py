"""
Graph validation utilities for weather-model-graphs.

This module provides comprehensive validation for graph integrity, including:
- Geometric consistency of 3D coordinates
- Metadata alignment with tensor shapes
- Connectivity health checks
- Hierarchical integrity for multi-level meshes
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import torch
import networkx as nx

try:
    import torch_geometric as pyg
    HAS_PYG = True
except ImportError:
    HAS_PYG = False
    warnings.warn("PyTorch Geometric not available, some validations may be limited")


def validate_geometric_consistency(mesh_positions: List[torch.Tensor], tolerance: float = 1e-6) -> bool:
    """
    Check that all 3D Cartesian coordinates sit on a unit sphere (L2 norm ≈1.0).

    Parameters
    ----------
    mesh_positions : List[torch.Tensor]
        List of mesh position tensors, one per level for hierarchical graphs.
    tolerance : float
        Tolerance for norm deviation from 1.0.

    Returns
    -------
    bool
        True if all positions are on unit sphere within tolerance.
    """
    all_valid = True
    for level, pos in enumerate(mesh_positions):
        if pos.shape[1] < 3:
            warnings.warn(f"Level {level}: Expected at least 3D coordinates, got {pos.shape[1]}D")
            all_valid = False
            continue

        # Only check the first 3 dimensions (x, y, z coordinates)
        coords = pos[:, :3]
        norms = torch.norm(coords, dim=1)
        deviations = torch.abs(norms - 1.0)
        max_deviation = deviations.max().item()

        if max_deviation > tolerance:
            warnings.warn(
                f"Level {level}: Max norm deviation from 1.0 is {max_deviation:.2e} "
                f"(tolerance: {tolerance}). "
                f"Found {torch.sum(deviations > tolerance).item()} invalid positions."
            )
            all_valid = False

    return all_valid


def validate_metadata_alignment(metadata_path: str, graph_tensors: Dict[str, Any]) -> bool:
    """
    Cross-reference metadata.json specs with actual tensor shapes in .pt files.

    Parameters
    ----------
    metadata_path : str
        Path to metadata.json file.
    graph_tensors : Dict[str, Any]
        Dictionary of loaded graph tensors.

    Returns
    -------
    bool
        True if shapes align with metadata specs.
    """
    try:
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        warnings.warn(f"Metadata file not found: {metadata_path}")
        return False

    all_valid = True

    # Check hierarchy
    expected_levels = metadata.get('hierarchy', {}).get('levels', 1)
    actual_levels = len(graph_tensors.get('m2m_edge_index', []))

    if actual_levels != expected_levels:
        warnings.warn(
            f"Hierarchy levels mismatch: expected {expected_levels}, got {actual_levels}"
        )
        all_valid = False

    # Check edge features
    edge_specs = metadata.get('edge_features', {})
    for edge_type, expected_features in edge_specs.items():
        if edge_type not in graph_tensors:
            warnings.warn(f"Missing edge type in tensors: {edge_type}")
            all_valid = False
            continue

        features_tensor = graph_tensors.get(f'{edge_type}_features')
        if features_tensor is None:
            warnings.warn(f"Missing features for edge type: {edge_type}")
            all_valid = False
            continue

        # For hierarchical, features is a list
        if isinstance(features_tensor, list):
            for level, feat in enumerate(features_tensor):
                if feat.shape[1] != len(expected_features):
                    warnings.warn(
                        f"{edge_type} level {level}: expected {len(expected_features)} features, "
                        f"got {feat.shape[1]}"
                    )
                    all_valid = False
        else:
            if features_tensor.shape[1] != len(expected_features):
                warnings.warn(
                    f"{edge_type}: expected {len(expected_features)} features, "
                    f"got {features_tensor.shape[1]}"
                )
                all_valid = False

    return all_valid


def validate_connectivity_health(graph_tensors: Dict[str, Any]) -> bool:
    """
    Identify isolated nodes or "dead-end" components that could break message passing.

    Uses PyTorch operations for efficiency with large graphs.

    Parameters
    ----------
    graph_tensors : Dict[str, Any]
        Dictionary of loaded graph tensors.

    Returns
    -------
    bool
        True if no connectivity issues found.
    """
    all_valid = True

    # Check mesh-to-mesh connectivity
    m2m_edge_index = graph_tensors.get('m2m_edge_index', [])
    if isinstance(m2m_edge_index, list) and m2m_edge_index:
        for level, edges in enumerate(m2m_edge_index):
            if hasattr(edges, 'shape') and edges.shape[1] > 0:
                # Get unique nodes in this level's edges
                present_nodes = torch.unique(edges.flatten())
                # Estimate total nodes (approximate, since we don't have exact count)
                max_node_idx = edges.max().item()
                if len(present_nodes) < max_node_idx + 1:
                    warnings.warn(
                        f"M2M level {level}: Found {len(present_nodes)} connected nodes "
                        f"out of expected ~{max_node_idx + 1}"
                    )
                    all_valid = False

    # Check grid-to-mesh connectivity
    g2m_edge_index = graph_tensors.get('g2m_edge_index')
    if g2m_edge_index is not None and hasattr(g2m_edge_index, 'shape'):
        if g2m_edge_index.shape[1] > 0:
            # Grid nodes are sources (row 0), mesh nodes are destinations (row 1)
            grid_nodes = torch.unique(g2m_edge_index[0])
            mesh_nodes = torch.unique(g2m_edge_index[1])

            # Check for isolated grid nodes (no outgoing g2m edges)
            # This is more complex to check efficiently without full graph reconstruction
            # For now, just warn if no g2m edges at all
            if len(grid_nodes) == 0:
                warnings.warn("No grid-to-mesh connections found")
                all_valid = False

    return all_valid


def validate_hierarchical_integrity(graph_tensors: Dict[str, Any]) -> bool:
    """
    Validate that inter-level edges correctly map between Li and Li+1 for multi-level meshes.

    Parameters
    ----------
    graph_tensors : Dict[str, Any]
        Dictionary of loaded graph tensors.

    Returns
    -------
    bool
        True if hierarchical structure is valid.
    """
    mesh_up_edge_index = graph_tensors.get('mesh_up_edge_index')
    mesh_down_edge_index = graph_tensors.get('mesh_down_edge_index')

    if mesh_up_edge_index is None or mesh_down_edge_index is None:
        # Not hierarchical, skip
        return True

    all_valid = True

    # Convert to numpy for easier analysis
    if hasattr(mesh_up_edge_index, 'cpu'):
        mesh_up = mesh_up_edge_index.cpu().numpy()
        mesh_down = mesh_down_edge_index.cpu().numpy()
    else:
        mesh_up = mesh_up_edge_index
        mesh_down = mesh_down_edge_index

    # Basic shape checks
    if mesh_up.shape[0] != 2 or mesh_down.shape[0] != 2:
        warnings.warn("Inter-level edge indices should have shape (2, N_edges)")
        all_valid = False

    # Check that up and down are inverses (bidirectional connectivity)
    # This is a simplified check - in practice, the mapping should be proper
    up_set = set(zip(mesh_up[0], mesh_up[1]))
    down_set = set(zip(mesh_down[0], mesh_down[1]))

    # Check for symmetry (up and down should connect same node pairs)
    symmetric = up_set == down_set
    if not symmetric:
        warnings.warn("Inter-level edges are not symmetric between up and down")
        all_valid = False

    # Check for isolated nodes in inter-level connectivity
    all_nodes_in_up = set(mesh_up.flatten())
    all_nodes_in_down = set(mesh_down.flatten())

    if all_nodes_in_up != all_nodes_in_down:
        warnings.warn("Node sets differ between up and down inter-level edges")
        all_valid = False

    return all_valid


def validate_graph_directory(graph_dir: str, metadata_path: str = None) -> Dict[str, bool]:
    """
    Run all validations on a graph directory.

    Parameters
    ----------
    graph_dir : str
        Path to directory containing graph .pt files.
    metadata_path : str, optional
        Path to metadata.json file. If None, looks for metadata.json in graph_dir.

    Returns
    -------
    Dict[str, bool]
        Validation results for each check.
    """
    if metadata_path is None:
        metadata_path = os.path.join(graph_dir, 'metadata.json')

    # Load graph tensors (simplified version of load_graph from neural-lam)
    graph_tensors = {}
    required_files = [
        'mesh_features.pt',
        'm2m_edge_index.pt',
        'g2m_edge_index.pt',
        'm2g_edge_index.pt',
        'm2m_features.pt',
        'g2m_features.pt',
        'm2g_features.pt'
    ]

    for filename in required_files:
        filepath = os.path.join(graph_dir, filename)
        if os.path.exists(filepath):
            try:
                graph_tensors[filename.replace('.pt', '')] = torch.load(filepath, weights_only=True)
            except Exception as e:
                warnings.warn(f"Failed to load {filename}: {e}")

    # Optional hierarchical files
    hierarchical_files = [
        'mesh_up_edge_index.pt',
        'mesh_down_edge_index.pt',
        'mesh_up_features.pt',
        'mesh_down_features.pt'
    ]

    for filename in hierarchical_files:
        filepath = os.path.join(graph_dir, filename)
        if os.path.exists(filepath):
            try:
                graph_tensors[filename.replace('.pt', '')] = torch.load(filepath, weights_only=True)
            except Exception as e:
                warnings.warn(f"Failed to load {filename}: {e}")

    results = {}

    # Geometric consistency
    mesh_features = graph_tensors.get('mesh_features', [])
    results['geometric_consistency'] = validate_geometric_consistency(mesh_features)

    # Metadata alignment
    if os.path.exists(metadata_path):
        results['metadata_alignment'] = validate_metadata_alignment(metadata_path, graph_tensors)
    else:
        warnings.warn(f"Metadata file not found: {metadata_path}")
        results['metadata_alignment'] = False

    # Connectivity health
    results['connectivity_health'] = validate_connectivity_health(graph_tensors)

    # Hierarchical integrity
    results['hierarchical_integrity'] = validate_hierarchical_integrity(graph_tensors)

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate weather model graphs")
    parser.add_argument("graph_dir", help="Directory containing graph .pt files")
    parser.add_argument("--metadata", help="Path to metadata.json file")

    args = parser.parse_args()

    results = validate_graph_directory(args.graph_dir, args.metadata)

    print("Validation Results:")
    for check, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {check}: {status}")

    all_passed = all(results.values())
    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'}")