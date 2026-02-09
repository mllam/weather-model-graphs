from __future__ import annotations

from typing import TYPE_CHECKING, Any, Mapping

import networkx
import numpy as np

if TYPE_CHECKING:
    import xarray as xr


def graph_from_icon_grid(
    dataset: Mapping[str, Any] | "xr.Dataset",
) -> networkx.Graph:
    """
    Create a networkx.Graph from an ICON grid dataset.

    Nodes correspond to cell centers with coordinates (clat, clon). Edges are
    created by enumerating neighbor_cell_index (nv = 3 for triangular cells).

    Parameters
    ----------
    dataset : xarray.Dataset or mapping
        Must provide variables "clat", "clon", and "neighbor_cell_index".

    Returns
    -------
    networkx.Graph
        Graph with node attributes "clat", "clon", and "pos".
    """

    clat = np.asarray(dataset["clat"])
    clon = np.asarray(dataset["clon"])
    neighbor = np.asarray(dataset["neighbor_cell_index"])

    if clat.ndim != 1 or clon.ndim != 1:
        raise ValueError("'clat' and 'clon' must be 1D arrays of cell centers.")

    if neighbor.ndim != 2:
        raise ValueError("'neighbor_cell_index' must be a 2D array [cell, nv].")

    n_cells = clat.shape[0]
    if clon.shape[0] != n_cells:
        raise ValueError("'clat' and 'clon' must have the same length.")

    if neighbor.shape[0] != n_cells:
        if neighbor.shape[1] == n_cells:
            neighbor = neighbor.T
        else:
            raise ValueError(
                "First dimension of 'neighbor_cell_index' must match number of cells."
            )

    neighbor = neighbor.astype(np.int64, copy=False)
    neighbor_min = neighbor.min()
    neighbor_max = neighbor.max()

    # ICON grids commonly store neighbors as 1-based indices with 0 as missing.
    if neighbor_max == n_cells:
        neighbor = neighbor - 1
        neighbor_min = neighbor_min - 1

    G = networkx.Graph()

    for i in range(n_cells):
        G.add_node(
            i,
            clat=float(clat[i]),
            clon=float(clon[i]),
            pos=(float(clon[i]), float(clat[i])),
        )

    for i in range(n_cells):
        for j in neighbor[i]:
            if j < 0 or j >= n_cells:
                continue
            if j == i:
                continue
            G.add_edge(i, int(j))

    return G
