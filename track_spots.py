from pathlib import Path
import pickle

import h5py
import numpy as np
import yaml

from hexrd import imageseries
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument import HEDMInstrument
from hexrd.material.material import load_materials_hdf5

from assign_spots import (
    assign_spots_to_hkls,
    chunk_spots_into_subpanels,
    combine_spots,
    in_range,
    track_spots,
)
from spot_finder import SpotFinder
from spot_tracker import TrackedSpot
from write_spots import write_spots

# Whether we should reuse the existing spots file. If we change
# anything that would change spot detection, this should be disabled.
use_existing_spots_file = True

# Whether we should, at the very end, create comparison plots between
# our spot detection and `pull_spots()` (for spots that only occur
# on one frame)
plot_comparison = False

# Filenames
instr_file = 'dual_dexelas_composite.yml'
image_files = {
    'ff1': 'mruby-0129_000004_ff1_000012-cachefile.npz',
    'ff2': 'mruby-0129_000004_ff2_000012-cachefile.npz',
}
materials_file = 'ruby.h5'
grain_params_file = 'grain_params.npy'
spots_filename = Path('spots_file.h5')

# Load the instrument
with open(instr_file, 'r') as rf:
    conf = yaml.safe_load(rf)

instr = HEDMInstrument(conf)

# Load in the raw imageseries for each Dexela detector
raw_ims_dict = {}
for k, filename in image_files.items():
    raw_ims_dict[k] = imageseries.open(filename, 'frame-cache')

# Grab the number of images and the omega period
num_images = len(raw_ims_dict[k])
omegas = raw_ims_dict[k].metadata['omega']
omega_period = np.radians((omegas[0][0], omegas[-1][1]))
eta_period = (-np.pi, np.pi)

# Just assume the whole eta and omega ranges are used
eta_ranges = [eta_period]
omega_ranges = [omega_period]

# Break up the imageseries into their subpanels
ims_dict = {}
for det_key, panel in instr.detectors.items():
    ops = [
        ('rectangle', panel.roi),
    ]

    raw_key = det_key[:3]
    ims = raw_ims_dict[raw_key]
    ims_dict[det_key] = ProcessedImageSeries(ims, ops)

# Load the material
with h5py.File(materials_file, 'r') as rf:
    material = load_materials_hdf5(rf)['ruby']

plane_data = material.planeData

# Load the known grain parameters
grain_params = np.load(grain_params_file)

# Create and write the spots file
if not use_existing_spots_file or not spots_filename.exists():
    finder = SpotFinder(min_area=2)
    with h5py.File(spots_filename, 'w') as f:
        write_spots(raw_ims_dict, finder, f)

TrackSpotsOutputType = dict[str, dict[int, list[TrackedSpot]]]

# Track, combine, and chunk spots into subpanels
tracked_spots = track_spots(spots_filename, num_images, list(raw_ims_dict))
spot_arrays = combine_spots(tracked_spots, omegas)
spot_arrays = chunk_spots_into_subpanels(spot_arrays, instr)

# Now simulate the spots
simulated_results = instr.simulate_rotation_series(
    plane_data,
    grain_params,
    eta_ranges,
    omega_ranges,
    omega_period,
)

# Set tolerances for tth, eta, and omega
tth_tol = np.radians(0.25)
eta_tol = np.radians(1.0)
ome_tol = np.radians(1.5)
tolerances = np.array([tth_tol, eta_tol, ome_tol])

# Now assign spots to HKLs
assigned_spots = assign_spots_to_hkls(
    spot_arrays,
    instr,
    simulated_results,
    grain_params,
    eta_period,
    tolerances,
)

# Compare with output from hexrdgui pull_spots()
# The `pull_spots()` output is the 'reference output'
with open('spots_data_dict_from_hexrdgui.pkl', 'rb') as rf:
    ref_spots_dict = pickle.load(rf)

# Keep track of which HKLs from `pull_spots()` were matched or
# unmatched.
num_matched = 0
num_unmatched = 0

# Keep track of the distances too
distances = []
ome_diffs = []

matched_spots = {}
for grain_id, (ref_completeness, ref_grain_spots) in ref_spots_dict.items():
    for det_key, ref_det_spots in ref_grain_spots.items():
        # Grab the measured spots
        meas_spots = assigned_spots[det_key][grain_id]

        # Keep track of which spots were assigned and which were not
        matched_spots.setdefault(det_key, {})
        these_matched_spots = np.zeros(len(meas_spots['hkls']), dtype=bool)
        matched_spots[det_key][grain_id] = these_matched_spots

        for ref_spot in ref_det_spots:
            hkl = ref_spot[2]
            ref_sum_int = ref_spot[3]
            ref_max_int = ref_spot[4]
            ref_meas_angs = ref_spot[6]
            ref_omega = ref_meas_angs[2]
            ref_meas_xy = ref_spot[7]

            if np.any(np.isnan(ref_meas_xy)):
                # This is not a real spot...
                # I don't know why `pull_spots()` sometimes does this...
                continue

            matching_idx = -1
            for row in range(len(meas_spots['hkls'])):
                if np.array_equal(meas_spots['hkls'][row], hkl):
                    # Found a matching HKL!
                    matching_idx = row
                    break

            if matching_idx == -1:
                print(
                    f'Warning! Did not find a match on {det_key} for HKL: '
                    f'{hkl}'
                )
                num_unmatched += 1
                continue

            # Compute the distance between the reference measured xy and
            # our own measured xy
            distance = np.sqrt(
                (
                    (ref_meas_xy - meas_spots['meas_xys'][matching_idx][:2])
                    ** 2
                ).sum()
            )
            distances.append(distance)

            # Compute the difference in omega between the reference and ours
            ome_diff = abs(ref_omega - meas_spots['meas_xys'][matching_idx][2])
            ome_diffs.append(ome_diff)

            # Indicate this spot was assigned
            these_matched_spots[matching_idx] = True
            num_matched += 1


max_omega_diff = max(ome_diffs)
max_distance = max(distances)
percent_found = num_matched / (num_matched + num_unmatched) * 100
print(
    'Percentage of spots that `pull_spots()` found that we also found:',
    f'{percent_found:.2f}%',
)

print(f'Mean distance (xy): {np.mean(distances):.4f}')
print(f'Max distance (xy): {max_distance:.4f}')
print(f'Mean omega diff (degrees): {np.degrees(np.mean(ome_diffs)):.4f}')
print(f'Max omega diff (degrees): {np.degrees(max_omega_diff):.4f}')
num_extra_hkls = 0
for det_key, det_assignments in matched_spots.items():
    for grain_id, grain_assignments in det_assignments.items():
        if np.any(~grain_assignments):
            extra_hkls = assigned_spots[det_key][grain_id]['hkls'][
                ~grain_assignments
            ]
            num_extra_hkls += len(extra_hkls)
            # print('Extra HKLs:', extra_hkls)

print(
    'Number of HKLs we found that `pull_spots()` did not find:', num_extra_hkls
)

if not plot_comparison:
    raise SystemExit

import matplotlib.pyplot as plt  # noqa

for det_key, meas_spots in assigned_spots.items():
    panel = instr.detectors[det_key]

    all_meas_xys = []
    all_ref_xys = []
    for grain_id, grain_spots in meas_spots.items():
        meas_xys = grain_spots['meas_xys']
        num_frames = grain_spots['num_frames']
        hkls = grain_spots['hkls']

        # Only keep spots that are on one frame so we can visualize them
        # properly
        keep = num_frames == 1
        meas_xys = meas_xys[keep]
        hkls = hkls[keep]

        # Now loop over reference spots and only keep HKLs that we kept
        # Keep them in the same order as the measured HKLs
        ref_spots = ref_spots_dict[grain_id][1][det_key]
        ref_meas_xys = []
        for hkl in hkls:
            found = False
            for ref_spot in ref_spots:
                ref_hkl = ref_spot[2]
                if np.array_equal(hkl, ref_hkl):
                    ref_meas_xy = ref_spot[7]
                    ref_meas_xys.append(ref_meas_xy)
                    found = True
                    break

            if not found:
                ref_meas_xys.append((np.nan, np.nan))

        all_meas_xys.append(meas_xys)
        all_ref_xys.append(np.array(ref_meas_xys))

    all_meas_xys = np.vstack(all_meas_xys)
    all_ref_xys = np.vstack(all_ref_xys)

    assert len(all_meas_xys) == len(all_ref_xys)

    for i in range(num_images):
        # Only keep spots that are on this frame
        keep_xys = in_range(np.degrees(all_meas_xys[:, 2]), omegas[i])
        kept_meas_xys = all_meas_xys[keep_xys]
        kept_ref_xys = all_ref_xys[keep_xys]

        if kept_meas_xys.size == 0:
            continue

        # Convert them to pixel coordinates
        meas_ij = panel.cartToPixel(
            kept_meas_xys[:, :2],
            apply_distortion=True,
        )[:, [1, 0]]
        ref_ij = panel.cartToPixel(
            kept_ref_xys,
            apply_distortion=True,
        )[:, [1, 0]]

        # Now get the image
        img = ims_dict[det_key][i]

        plt.title(f'{det_key} frame {i}')
        plt.imshow(img, vmin=0, vmax=6000, origin='lower', cmap='Grays')
        plt.scatter(*meas_ij.T, marker='x', c='red', label='meas')
        plt.scatter(*ref_ij.T, marker='+', c='blue', label='ref')
        plt.legend()
        plt.show()
