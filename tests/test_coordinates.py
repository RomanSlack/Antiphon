"""Coordinate-convention correctness: y=0 is the street centerline."""

import numpy as np

from antiphon.simulation import (
    AcousticField,
    NoiseSource,
    SpeakerArray,
    UrbanGeometry,
)


def test_to_grid_centerline_convention():
    geo = UrbanGeometry()
    # Row 0 is the outer edge of the lower building
    assert geo.to_grid(0.0, geo.y_min) == (0, 0)
    # y=0 (centerline) maps to the middle row
    _, iy = geo.to_grid(0.0, 0.0)
    assert iy == geo.ny // 2
    # Street edges map to the building boundaries in the mask
    _, iy_edge = geo.to_grid(0.0, -geo.street_width / 2)
    assert geo.mask[0, iy_edge] == 0
    assert geo.mask[0, iy_edge - 1] == 1


def test_field_grid_matches_to_grid():
    geo = UrbanGeometry()
    field = AcousticField(geo)
    # The field's physical y coordinates span the same domain as to_grid
    assert field.y_coords[0] == geo.y_min
    np.testing.assert_allclose(field.y_coords[-1], geo.y_max - geo.res)
    # A physical point maps to the same grid cell in both systems
    ix, iy = geo.to_grid(12.3, -4.5)
    assert abs(field.x_coords[ix] - 12.3) < geo.res
    assert abs(field.y_coords[iy] - (-4.5)) < geo.res


def test_speakers_inside_street():
    geo = UrbanGeometry()
    field = AcousticField(geo)
    for side in ('left', 'right'):
        arr = SpeakerArray(6, side, geo)
        for (sx, sy) in arr.positions:
            ix, iy = geo.to_grid(sx, sy)
            assert geo.mask[ix, iy] == 0, f'speaker at ({sx},{sy}) inside building'
            assert field.y_coords[0] < sy < field.y_coords[-1]


def test_centered_source_field_is_symmetric():
    """A source on the centerline must produce a y-symmetric field
    (both image-source walls are now real walls, equidistant)."""
    geo = UrbanGeometry()
    field = AcousticField(geo)
    noise = NoiseSource(x=geo.street_length / 2, y=0.0, frequency=200)
    p = field.compute_pressure(noise, [], t=1e-3, mode='off')

    for y_probe in (2.0, 5.0, 8.0):
        ix, iy_pos = geo.to_grid(10.0, y_probe)
        _, iy_neg = geo.to_grid(10.0, -y_probe)
        np.testing.assert_allclose(p[ix, iy_pos], p[ix, iy_neg], rtol=0.02)
