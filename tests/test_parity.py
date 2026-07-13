"""Parity with the reference script (refs/urban_anc_simulation.py).

The initial refactor was verified bit-identical against the reference. The
coordinate-system fix (street centerline is now y=0 everywhere) intentionally
changed physical-coordinate behavior: the reference placed the field grid's
y=0 at the street's bottom edge while speakers and image-source walls assumed
a centerline origin, so its speakers sat outside the simulated domain and its
reflections used wrong wall positions. Index-space geometry is unchanged, so
parity is still enforced there.
"""

import numpy as np

from antiphon.simulation import UrbanGeometry


def test_geometry_index_space_parity(ref_sim):
    ref_geo = ref_sim.UrbanGeometry()
    new_geo = UrbanGeometry()

    assert (ref_geo.nx, ref_geo.ny) == (new_geo.nx, new_geo.ny)
    np.testing.assert_array_equal(ref_geo.mask, new_geo.mask)
    np.testing.assert_array_equal(ref_geo.sidewalk_mask, new_geo.sidewalk_mask)
    np.testing.assert_array_equal(ref_geo.ped_zone, new_geo.ped_zone)
