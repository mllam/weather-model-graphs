"""
Spatial indexing for metric-aware nearest-neighbour and radius queries.

Provides :class:`SpatialCoordinateValuesSelector`, which wraps a ball-tree to
deliver k-nearest-neighbour and radius lookups using either the Euclidean or the
Haversine distance metric.  The correct metric is chosen automatically when the
object is created via the :meth:`SpatialCoordinateValuesSelector.for_crs` class
method.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np

try:
    from sklearn.neighbors import BallTree as _BallTree

    _HAS_SKLEARN = True
except ImportError:  # pragma: no cover
    _HAS_SKLEARN = False


class SpatialCoordinateValuesSelector:
    """
    Metric-aware spatial index for selecting coordinate values by proximity.

    Wraps a ball-tree to provide fast k-nearest-neighbour and radius queries.
    The tree is built lazily — it is only constructed when a tree-based query
    (:meth:`k_nearest_to` or :meth:`within_radius`) is first called.
    :meth:`distance_between` performs direct computation without the tree,
    so callers that only need pairwise distances (e.g. ``create_directed_mesh_graph``)
    never pay the cost of building the tree.

    Two distance metrics are supported:

    * ``"euclidean"`` – standard Cartesian distance, appropriate for projected
      coordinate systems (e.g. Lambert Conformal, UTM).
    * ``"haversine"`` – great-circle distance on a sphere, appropriate for
      geographic coordinate systems expressed as longitude/latitude in degrees
      (e.g. PlateCarree).

    Parameters
    ----------
    distance_metric : {'euclidean', 'haversine'}
        Distance metric to use for the underlying ball-tree.
    coords : np.ndarray, shape (N, 2)
        Coordinate array.  For ``"euclidean"`` these are arbitrary Cartesian
        (x, y) values.  For ``"haversine"`` these must be **longitude/latitude
        in degrees** (first column longitude, second column latitude).

    Raises
    ------
    ValueError
        If *distance_metric* is not ``"euclidean"`` or ``"haversine"``.
    ImportError
        If *distance_metric* is ``"haversine"`` and ``scikit-learn`` is not
        installed.

    Examples
    --------
    Euclidean (projected CRS):

    >>> import numpy as np
    >>> coords = np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]])
    >>> sel = SpatialCoordinateValuesSelector("euclidean", coords)
    >>> idxs, dists = sel.k_nearest_to([1.5, 0.0], k=2)
    >>> idxs.tolist()
    [2, 1]

    Haversine (geographic CRS, lon/lat degrees):

    >>> coords_geo = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]])
    >>> sel_geo = SpatialCoordinateValuesSelector("haversine", coords_geo)
    >>> idxs, dists = sel_geo.k_nearest_to([5.0, 0.0], k=2)
    """

    def __init__(self, distance_metric: str, coords: np.ndarray) -> None:
        _VALID_METRICS = ("euclidean", "haversine")
        if distance_metric not in _VALID_METRICS:
            raise ValueError(
                f"distance_metric must be one of {_VALID_METRICS!r}, "
                f"got {distance_metric!r}."
            )

        if distance_metric == "haversine" and not _HAS_SKLEARN:
            raise ImportError(
                "scikit-learn is required for the 'haversine' distance metric. "
                "Install it with:  pip install scikit-learn"
            )

        self.distance_metric: str = distance_metric
        self._coords: np.ndarray = np.asarray(coords, dtype=float)
        self._tree: _BallTree | None = None

    # Private lazy tree constructor
    def _get_tree(self) -> _BallTree:
        """Build and cache the ball-tree on first call."""
        if self._tree is not None:
            return self._tree
        if self.distance_metric == "haversine":
            tree_coords = np.deg2rad(self._coords[:, ::-1])
            self._tree = _BallTree(tree_coords, metric="haversine")
        else:
            self._tree = _BallTree(self._coords, metric="euclidean")
        return self._tree

    # Public query methods
    def k_nearest_to(self, point: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the *k* nearest coordinate values to *point*.

        Parameters
        ----------
        point : array-like, shape (2,)
            Query point.  For ``"euclidean"`` this is a Cartesian (x, y)
            coordinate; for ``"haversine"`` this is a (longitude, latitude)
            pair in degrees.
        k : int
            Number of nearest neighbours to return.

        Returns
        -------
        indices : np.ndarray, shape (k,)
            Indices into the original *coords* array (passed to ``__init__``)
            of the *k* nearest neighbours, ordered by increasing distance.
        distances : np.ndarray, shape (k,)
            Corresponding distances.  For ``"euclidean"`` these are in the same
            units as *coords*; for ``"haversine"`` these are in **degrees**.
        """
        tree = self._get_tree()
        tree_point = self._prepare_query_point(point)
        raw_dists, raw_idxs = tree.query(tree_point, k=k)
        indices = raw_idxs.flatten()
        distances = (
            np.rad2deg(raw_dists.flatten())
            if self.distance_metric == "haversine"
            else raw_dists.flatten()
        )
        return indices, distances

    def within_radius(
        self, point: np.ndarray, radius: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all coordinate values within *radius* of *point*.

        Parameters
        ----------
        point : array-like, shape (2,)
            Query point (same coordinate convention as for :meth:`k_nearest_to`).
        radius : float
            Search radius.  For ``"euclidean"`` this is in the same units as
            *coords*; for ``"haversine"`` this is in **degrees**.

        Returns
        -------
        indices : np.ndarray
            Indices into the original *coords* array of all neighbours within
            *radius*, **unsorted**.
        distances : np.ndarray
            Distances to each returned neighbour.  For ``"euclidean"`` these
            are in the same units as *coords*; for ``"haversine"`` these are
            in **degrees**.
        """
        tree = self._get_tree()
        tree_point = self._prepare_query_point(point)
        raw_idxs, raw_dists = tree.query_radius(
            tree_point,
            r=np.deg2rad(radius) if self.distance_metric == "haversine" else radius,
            return_distance=True,
        )
        indices = raw_idxs[0]
        distances = (
            np.rad2deg(raw_dists[0])
            if self.distance_metric == "haversine"
            else raw_dists[0]
        )
        return indices, distances

    # Factory class-method
    @classmethod
    def for_crs(
        cls,
        crs,
        coords: np.ndarray,
    ) -> "SpatialCoordinateValuesSelector":
        """
        Create a :class:`SpatialCoordinateValuesSelector` appropriate for *crs*.

        Inspects the ``is_geographic`` property of *crs* to choose the metric:

        * Geographic CRS (``crs.is_geographic is True``) → ``"haversine"``
        * Projected CRS → ``"euclidean"``

        Parameters
        ----------
        crs : cartopy.crs.CRS or pyproj.CRS
            Coordinate reference system of *coords*.  Must expose an
            ``is_geographic`` attribute (both *cartopy* and *pyproj* CRS
            objects do).
        coords : np.ndarray, shape (N, 2)
            Coordinate array in the given *crs*.

        Returns
        -------
        SpatialCoordinateValuesSelector
            Configured with the appropriate distance metric.

        Examples
        --------
        >>> import cartopy.crs as ccrs
        >>> import numpy as np
        >>> coords = np.column_stack([np.linspace(-10, 10, 50),
        ...                           np.linspace(50, 60, 50)])
        >>> sel = SpatialCoordinateValuesSelector.for_crs(ccrs.PlateCarree(), coords)
        >>> sel.distance_metric
        'haversine'
        >>> sel2 = SpatialCoordinateValuesSelector.for_crs(ccrs.LambertConformal(), coords)
        >>> sel2.distance_metric
        'euclidean'
        """
        is_geographic = getattr(crs, "is_geographic", False)
        # pyproj CRS exposes is_geographic as a bool property; cartopy CRS does too.
        metric = "haversine" if is_geographic else "euclidean"
        return cls(metric, coords)

    # Public pairwise distance method
    def distance_between(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        """
        Compute the distance between two points using the configured metric.

        Parameters
        ----------
        point_a : array-like, shape (2,)
            First point. Same coordinate convention as :meth:`k_nearest_to`.
        point_b : array-like, shape (2,)
            Second point. Same coordinate convention as :meth:`k_nearest_to`.

        Returns
        -------
        float
            Distance between *point_a* and *point_b*. For ``"euclidean"``
            this is in the same units as *coords*; for ``"haversine"``
            this is in degrees.
        """
        pt_a = self._prepare_query_point(point_a)
        pt_b = self._prepare_query_point(point_b)
        if self.distance_metric == "euclidean":
            return float(np.linalg.norm(pt_a - pt_b))
        elif self.distance_metric == "haversine":
            # pt_a and pt_b are [lat_rad, lon_rad] shaped (1, 2)
            lat1, lon1 = pt_a[0, 0], pt_a[0, 1]
            lat2, lon2 = pt_b[0, 0], pt_b[0, 1]
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = (
                np.sin(dlat / 2) ** 2
                + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
            )
            return float(np.rad2deg(2 * np.arcsin(np.sqrt(a))))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric!r}")

    # Private helpers
    def _prepare_query_point(self, point: np.ndarray) -> np.ndarray:
        """Convert a query point to the internal representation used by the tree."""
        pt = np.asarray(point, dtype=float).reshape(1, 2)
        if self.distance_metric == "haversine":
            # swap [lon, lat] → [lat, lon] and convert to radians
            return np.deg2rad(pt[:, ::-1])
        return pt
