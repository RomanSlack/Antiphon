"""
Urban Active Noise Cancellation — 2D Wave Propagation Simulation
================================================================

Simulates sound propagation in a 2D urban canyon (top-down street view)
with active noise cancellation via speaker arrays.

Usage:
    python scripts/run_simulation.py                    # Run full simulation
    python scripts/run_simulation.py --freq 200         # Set noise frequency (Hz)
    python scripts/run_simulation.py --speakers 8       # Set speakers per side
    python scripts/run_simulation.py --sweep            # Frequency sweep analysis
    python scripts/run_simulation.py --animate          # Generate animation frames
    python scripts/run_simulation.py --save results.png # Save output figure

Author: Third Axis AI Consulting / 316 Group
"""

import argparse
import os

from antiphon.simulation import AcousticField, NoiseSource, SpeakerArray, UrbanGeometry
from antiphon.viz import frequency_sweep, plot_results, render_animation_frames

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'docs', 'figures')


def main():
    parser = argparse.ArgumentParser(description='Urban ANC Simulation')
    parser.add_argument('--freq', type=float, default=200,
                        help='Noise frequency in Hz (default: 200)')
    parser.add_argument('--speakers', type=int, default=6,
                        help='Number of speakers per side (default: 6)')
    parser.add_argument('--save', type=str, default=None,
                        help='Output file path (default: docs/figures/)')
    parser.add_argument('--sweep', action='store_true',
                        help='Run frequency sweep analysis')
    parser.add_argument('--animate', action='store_true',
                        help='Generate animation frames')
    args = parser.parse_args()

    print("=" * 60)
    print("Urban Active Noise Cancellation Simulation")
    print("=" * 60)

    os.makedirs(FIGURES_DIR, exist_ok=True)

    # Setup geometry
    geo = UrbanGeometry()
    print(f"\nGeometry: {geo.street_length}m x {geo.street_width}m street")
    print(f"Grid: {geo.nx} x {geo.ny} cells ({geo.res}m resolution)")

    # Setup noise source (center of street, at road level)
    noise = NoiseSource(x=geo.street_length / 2, y=0.0,
                        frequency=args.freq, amplitude=1.0)
    print(f"\nNoise source: {args.freq} Hz at ({noise.x}, {noise.y})m")
    print(f"Wavelength: {noise.wavelength:.2f}m")
    print(f"Theoretical quiet zone radius (λ/10): {noise.wavelength/10:.2f}m")

    # Setup speaker arrays
    spk_left = SpeakerArray(args.speakers, 'left', geo)
    spk_right = SpeakerArray(args.speakers, 'right', geo)
    speaker_arrays = [spk_left, spk_right]
    print(f"Speakers: {args.speakers} per side ({2*args.speakers} total)")

    # Setup acoustic field
    field = AcousticField(geo)

    if args.sweep:
        print("\n--- Frequency Sweep ---")
        sweep_path = args.save or os.path.join(FIGURES_DIR, 'anc_frequency_sweep.png')
        frequency_sweep(speaker_arrays, geo, field, save_path=sweep_path)
    else:
        print("\n--- Computing Sound Fields ---")
        results_path = args.save or os.path.join(FIGURES_DIR, 'anc_simulation_results.png')
        plot_results(field, noise, speaker_arrays, geo, results_path)

    if args.animate:
        print("\n--- Generating Animation Frames ---")
        render_animation_frames(field, noise, speaker_arrays, geo,
                                output_dir=os.path.join(FIGURES_DIR, 'anc_frames'))

    print("\nDone.")


if __name__ == '__main__':
    main()
