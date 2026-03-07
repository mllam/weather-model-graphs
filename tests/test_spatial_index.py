"""
Tests for :class:`weather_model_graphs.spatial.SpatialCoordinateValuesSelector`.

Covers:
- Initialisation (valid / invalid metric)
- Euclidean k-nearest-to and with_radius queries
- Haversine k-nearest-to and with_radius queries (distances in radians)
- Factory method SpatialCoordinateValuesSelector.for_crs()
- Warning emitted for rectilinear mesh + geographic CRS in create_all_graph_components
"""

import warnings

import cartopy.crs as ccrs
import numpy as np
import pyproj
import pytest
from loguru import logger

import weather_model_graphs as wmg
from weather_model_graphs.spatial import SpatialCoordinateValuesSelector


# Fixtures
@pytest.fixture()
def simple_euclidean_coords():
    """Five points on a horizontal line: x = 0, 1, 2, 3, 4; y = 0."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]])


@pytest.fixture()
def simple_geo_coords():
    """Five lon/lat points along the equator (y = 0 °), 0–40 ° longitude."""
    return np.array(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0], [40.0, 0.0]]
    )


# Initialisation
class TestInit:
    def test_euclidean_metric_stored(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        assert sel.distance_metric == "euclidean"

    def test_haversine_metric_stored(self, simple_geo_coords):
        sel = SpatialCoordinateValuesSelector("haversine", simple_geo_coords)
        assert sel.distance_metric == "haversine"

    def test_invalid_metric_raises(self, simple_euclidean_coords):
        with pytest.raises(ValueError, match="distance_metric must be one of"):
            SpatialCoordinateValuesSelector("manhattan", simple_euclidean_coords)

    def test_coords_stored_as_float_array(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        assert sel._coords.dtype == np.float64


# Euclidean – k_nearest_to
class TestEuclideanKNearest:
    def test_self_distance_is_zero(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, dists = sel.k_nearest_to([0.0, 0.0], k=1)
        assert idxs[0] == 0
        assert dists[0] == pytest.approx(0.0)

    def test_nearest_of_two(self):
        coords = np.array([[0.0, 0.0], [10.0, 0.0]])
        sel = SpatialCoordinateValuesSelector("euclidean", coords)
        idxs, dists = sel.k_nearest_to([3.0, 0.0], k=1)
        assert idxs[0] == 0  # [0,0] is closer than [10,0]

    def test_k_neighbours_returned(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, dists = sel.k_nearest_to([2.0, 0.0], k=3)
        assert len(idxs) == 3
        assert len(dists) == 3

    def test_distances_sorted_ascending(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, dists = sel.k_nearest_to([0.5, 0.0], k=3)
        assert list(dists) == sorted(dists)

    def test_known_euclidean_distance(self):
        # Two points: (0,0) and (3,4) – distance = 5
        coords = np.array([[0.0, 0.0], [3.0, 4.0]])
        sel = SpatialCoordinateValuesSelector("euclidean", coords)
        idxs, dists = sel.k_nearest_to([0.0, 0.0], k=2)
        assert dists[1] == pytest.approx(5.0)


# Euclidean – with_radius
class TestEuclideanWithRadius:
    def test_returns_points_within_radius(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, dists = sel.with_radius([2.0, 0.0], radius=1.5)
        # should include indices 1, 2, 3  (x=1, 2, 3)
        assert set(idxs) == {1, 2, 3}

    def test_excludes_points_beyond_radius(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, _ = sel.with_radius([2.0, 0.0], radius=0.5)
        assert set(idxs) == {2}

    def test_distances_within_radius(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, dists = sel.with_radius([2.0, 0.0], radius=1.5)
        # All returned distances must be ≤ radius
        assert all(d <= 1.5 + 1e-9 for d in dists)

    def test_zero_radius_returns_only_self(self, simple_euclidean_coords):
        sel = SpatialCoordinateValuesSelector("euclidean", simple_euclidean_coords)
        idxs, dists = sel.with_radius([1.0, 0.0], radius=0.0)
        assert set(idxs) == {1}
        assert dists[0] == pytest.approx(0.0)



# Haversine – k_nearest_to
class TestHaversineKNearest:
    def test_self_distance_is_zero(self, simple_geo_coords):
        sel = SpatialCoordinateValuesSelector("haversine", simple_geo_coords)
        idxs, dists = sel.k_nearest_to([0.0, 0.0], k=1)
        assert idxs[0] == 0
        assert dists[0] == pytest.approx(0.0, abs=1e-3)

    def test_distances_in_radians(self, simple_geo_coords):
        """10° longitude at equator is 10° * pi/180 radians."""
        sel = SpatialCoordinateValuesSelector("haversine", simple_geo_coords)
        idxs, dists = sel.k_nearest_to([0.0, 0.0], k=2)
        # nearest is self (0 rad), second is [10, 0] = deg2rad(10)
        expected_rad = np.deg2rad(10.0)
        assert dists[1] == pytest.approx(expected_rad, rel=1e-4)

    def test_distances_are_native_haversine_radians(self):
        """For geographic coords, haversine returns unit-sphere radians."""
        coords = np.array([[0.0, 0.0], [1.0, 0.0]])
        sel = SpatialCoordinateValuesSelector("haversine", coords)
        _, d_hav = sel.k_nearest_to([0.0, 0.0], k=2)
        assert d_hav[1] == pytest.approx(np.deg2rad(1.0), rel=1e-4)


# Haversine – with_radius
class TestHaversineWithRadius:
    def test_radius_in_radians_inclusive(self, simple_geo_coords):
        """A 0.2 rad radius from origin includes points at 0° and 10° lon."""
        sel = SpatialCoordinateValuesSelector("haversine", simple_geo_coords)
        radius_rad = 0.2
        idxs, dists = sel.with_radius([0.0, 0.0], radius=radius_rad)
        assert 0 in idxs  # self
        assert 1 in idxs  # 10° lon ≈ 0.1745 rad away

    def test_radius_in_radians_exclusive(self, simple_geo_coords):
        """A 0.1 rad radius from origin excludes the 10° point."""
        sel = SpatialCoordinateValuesSelector("haversine", simple_geo_coords)
        radius_rad = 0.1
        idxs, _ = sel.with_radius([0.0, 0.0], radius=radius_rad)
        assert set(idxs) == {0}



# Factory: for_crs
class TestForCrs:
    def test_geographic_crs_gives_haversine(self):
        coords = np.random.default_rng(0).random((10, 2))
        # pyproj.CRS('EPSG:4326') is a true geographic CRS: is_geographic=True
        sel = SpatialCoordinateValuesSelector.for_crs(pyproj.CRS("EPSG:4326"), coords)
        assert sel.distance_metric == "haversine"

    def test_projected_crs_gives_euclidean(self):
        coords = np.random.default_rng(0).random((10, 2)) * 1e6
        sel = SpatialCoordinateValuesSelector.for_crs(ccrs.LambertConformal(), coords)
        assert sel.distance_metric == "euclidean"

    def test_mollweide_projected_gives_euclidean(self):
        coords = np.random.default_rng(0).random((10, 2)) * 1e6
        sel = SpatialCoordinateValuesSelector.for_crs(ccrs.Mollweide(), coords)
        assert sel.distance_metric == "euclidean"

    def test_equivalent_to_manual_construction_euclidean(self):
        """for_crs on a projected CRS should produce the same results as the
        manually constructed euclidean selector."""
        rng = np.random.default_rng(1)
        coords = rng.random((20, 2)) * 1e5
        query = [5e4, 5e4]
        sel_factory = SpatialCoordinateValuesSelector.for_crs(ccrs.LambertConformal(), coords)
        sel_manual = SpatialCoordinateValuesSelector("euclidean", coords)
        idxs_f, dists_f = sel_factory.k_nearest_to(query, k=3)
        idxs_m, dists_m = sel_manual.k_nearest_to(query, k=3)
        np.testing.assert_array_equal(idxs_f, idxs_m)
        np.testing.assert_allclose(dists_f, dists_m)



# Integration: rectilinear + geographic warning
class TestRectilinearGeographicWarning:
    """
    When create_all_graph_components is called with a geographic graph_crs and
    a rectilinear m2m_connectivity, a UserWarning should be raised.
    """

    def _make_lonlat_coords(self, n=10):
        lon = np.linspace(-10.0, 10.0, n)
        lat = np.linspace(50.0, 60.0, n)
        lo, la = np.meshgrid(lon, lat)
        return np.column_stack([lo.ravel(), la.ravel()])

    @pytest.mark.parametrize("m2m", ["flat", "flat_multiscale", "hierarchical"])
    def test_warning_raised_for_geographic_crs(self, m2m):
        # Use a 30x30 grid over a ~29-degree domain so hierarchical can build >=2 levels
        lon = np.linspace(0.0, 29.0, 30)
        lat = np.linspace(45.0, 74.0, 30)
        lo, la = np.meshgrid(lon, lat)
        large_coords = np.column_stack([lo.ravel(), la.ravel()])

        # pyproj EPSG:4326 has is_geographic=True → triggers warning
        geo_crs = pyproj.CRS("EPSG:4326")

        kwargs = dict(
            coords=large_coords,
            m2m_connectivity=m2m,
            g2m_connectivity="nearest_neighbour",
            m2g_connectivity="nearest_neighbour",
            graph_crs=geo_crs,
        )
        if m2m == "flat":
            kwargs["m2m_connectivity_kwargs"] = dict(mesh_node_distance=3)
        elif m2m == "flat_multiscale":
            kwargs["m2m_connectivity_kwargs"] = dict(
                max_num_levels=2, mesh_node_distance=3, level_refinement_factor=3
            )
        elif m2m == "hierarchical":
            kwargs["m2m_connectivity_kwargs"] = dict(
                max_num_levels=2, mesh_node_distance=3, level_refinement_factor=3
            )
        warning_messages = []
        sink_id = logger.add(
            lambda msg: warning_messages.append(msg.record["message"]),
            level="WARNING",
        )
        try:
            wmg.create.create_all_graph_components(**kwargs)
        finally:
            logger.remove(sink_id)

        assert any("rectilinear" in message for message in warning_messages)

    def test_no_warning_for_projected_crs(self):
        """No UserWarning for a projected CRS."""
        # Use a small Cartesian grid (pretend it's in some projected CRS)
        coords = np.column_stack([
            np.linspace(0, 1e5, 20), np.linspace(0, 1e5, 20)
        ])

        class _FakeProjectedCRS:
            """Minimal projected CRS stub."""
            is_geographic = False

        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            # Should *not* raise
            wmg.create.create_all_graph_components(
                coords=coords,
                m2m_connectivity="flat",
                m2m_connectivity_kwargs=dict(mesh_node_distance=3000),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
                graph_crs=_FakeProjectedCRS(),
            )


# Integration: graph creation uses correct metric end-to-end
class TestIntegrationGraphCreation:
    """
    Smoke-test that graph creation completes without error when a geographic
    CRS is supplied, and that the haversine-based edge lengths are physically
    reasonable in radians for a ~10° domain.
    """

    def _make_lonlat_coords(self, n=8):
        lon = np.linspace(0.0, 9.0, n)
        lat = np.linspace(50.0, 59.0, n)
        lo, la = np.meshgrid(lon, lat)
        return np.column_stack([lo.ravel(), la.ravel()])

    def test_graph_created_with_geographic_crs(self):
        coords = self._make_lonlat_coords()
        # pyproj EPSG:4326 has is_geographic=True → haversine metric used,
        # and the rectilinear/geographic warning is logged.
        warning_messages = []
        sink_id = logger.add(
            lambda msg: warning_messages.append(msg.record["message"]),
            level="WARNING",
        )
        try:
            G = wmg.create.create_all_graph_components(
                coords=coords,
                m2m_connectivity="flat",
                m2m_connectivity_kwargs=dict(mesh_node_distance=3),
                g2m_connectivity="nearest_neighbour",
                m2g_connectivity="nearest_neighbour",
                graph_crs=pyproj.CRS("EPSG:4326"),
                return_components=False,
            )
        finally:
            logger.remove(sink_id)

        assert any("rectilinear" in message for message in warning_messages)
        # The g2m / m2g edges use haversine (distances in radians).
        # The m2m internal mesh edges still use Euclidean (degrees) because
        # create_single_level_2d_mesh_graph does not receive the CRS.
        g2m_m2g_lens = [
            d["len"]
            for _, _, d in G.edges(data=True)
            if d.get("component") in ("g2m", "m2g") and "len" in d
        ]
        assert len(g2m_m2g_lens) > 0, "Expected g2m/m2g edges with 'len' attribute"
        # For a ~9 degree domain, haversine edge lengths should be below ~0.2 rad
        # and clearly not degree-scale values.
        assert all(1e-4 < l < 0.5 for l in g2m_m2g_lens), (
            f"g2m/m2g edge lengths out of expected haversine range: "
            f"min={min(g2m_m2g_lens):.6f} rad, max={max(g2m_m2g_lens):.6f} rad"
        )
