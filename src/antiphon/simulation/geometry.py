"""Urban canyon geometry and physical constants."""

import numpy as np

C_SOUND = 343.0         # Speed of sound in air (m/s) at 20°C
RHO_AIR = 1.225         # Air density (kg/m³)
SAMPLE_RATE = 48000     # Audio sample rate (Hz), professional audio standard

# Urban geometry defaults (meters)
STREET_WIDTH = 20.0     # Distance between buildings
STREET_LENGTH = 40.0    # Length of simulated corridor
BUILDING_DEPTH = 5.0    # Depth of building walls (for reflections)
SIDEWALK_WIDTH = 3.0    # Pedestrian zone width on each side

GRID_RESOLUTION = 0.1   # meters per grid cell (λ/10 at 343 Hz)


class UrbanGeometry:
    """Defines the 2D urban canyon geometry with buildings and sidewalks.

    Coordinate convention: y = 0 is the street centerline. The street spans
    y in [-street_width/2, +street_width/2]; buildings extend building_depth
    beyond each street edge. x runs along the street from 0 to street_length.
    """

    def __init__(self, street_width=STREET_WIDTH, street_length=STREET_LENGTH,
                 building_depth=BUILDING_DEPTH, sidewalk_width=SIDEWALK_WIDTH,
                 resolution=GRID_RESOLUTION):
        self.street_width = street_width
        self.street_length = street_length
        self.building_depth = building_depth
        self.sidewalk_width = sidewalk_width
        self.res = resolution

        self.nx = int(street_length / resolution)
        self.ny = int((street_width + 2 * building_depth) / resolution)

        # Create geometry mask: 0 = air, 1 = building (rigid boundary)
        self.mask = np.zeros((self.nx, self.ny), dtype=np.float32)

        # Building walls
        bld_cells = int(building_depth / resolution)
        self.mask[:, :bld_cells] = 1.0              # Left building
        self.mask[:, -bld_cells:] = 1.0              # Right building

        # Sidewalk zone (for quiet zone metrics)
        street_start = bld_cells
        street_end = self.ny - bld_cells
        sw_cells = int(sidewalk_width / resolution)
        self.sidewalk_mask = np.zeros_like(self.mask)
        self.sidewalk_mask[:, street_start:street_start + sw_cells] = 1.0
        self.sidewalk_mask[:, street_end - sw_cells:street_end] = 1.0

        # Pedestrian zone (center of sidewalks, where we measure quiet zone)
        self.ped_zone = np.zeros_like(self.mask)
        ped_start_l = street_start + sw_cells // 4
        ped_end_l = street_start + 3 * sw_cells // 4
        ped_start_r = street_end - 3 * sw_cells // 4
        ped_end_r = street_end - sw_cells // 4
        self.ped_zone[:, ped_start_l:ped_end_l] = 1.0
        self.ped_zone[:, ped_start_r:ped_end_r] = 1.0

    @property
    def y_min(self):
        """Physical y of grid row 0 (outer edge of the lower building)."""
        return -(self.street_width / 2 + self.building_depth)

    @property
    def y_max(self):
        """Physical y of the last grid row (outer edge of the upper building)."""
        return self.street_width / 2 + self.building_depth

    def to_grid(self, x_m, y_m):
        """Convert physical coordinates (m) to grid indices."""
        return int(x_m / self.res), int((y_m - self.y_min) / self.res)
