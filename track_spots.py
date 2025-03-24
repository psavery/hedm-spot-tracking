import copy
from pathlib import Path
import pickle

import h5py
import numpy as np
import yaml

from hexrd import imageseries
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument import HEDMInstrument
from hexrd.material.material import load_materials_hdf5
from hexrd.rotations import mapAngle

from spot_finder import SpotFinder
from spot_tracker import SpotTracker, TrackedSpot

# Whether we should cache the tracked spots for testing...
# If we change anything that would change spot detection or tracking, this
# should be disabled.
cache_spots = True

# Filenames
instr_file = 'dual_dexelas_composite.yml'
image_files = {
    'ff1': 'mruby-0129_000004_ff1_000012-cachefile.npz',
    'ff2': 'mruby-0129_000004_ff2_000012-cachefile.npz',
}
materials_file = 'ruby.h5'
grain_params_file = 'grain_params.npy'

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

# Find and track the spots in the raw images
finder = SpotFinder()
spot_trackers = {
    'ff1': SpotTracker(),
    'ff2': SpotTracker(),
}
all_spots = {
    'ff1': {},
    'ff2': {},
}

if cache_spots and Path('spots_cached.pkl').exists():
    # Just load the cached results...
    with open('spots_cached.pkl', 'rb') as rf:
        all_spots = pickle.load(rf)
else:
    for frame_index in range(num_images):
        for det_key, ims in raw_ims_dict.items():
            # First, find spots
            img = ims[frame_index]
            spots = finder.find_spots(img)

            # Next, track spots
            tracker = spot_trackers[det_key]
            tracker.track_spots(spots, frame_index)

            # Now update our tracked list
            spot_dict = all_spots[det_key]
            for spot_id, spot in tracker.current_spots.items():
                spot_list = spot_dict.setdefault(spot_id, [])
                spot_list.append(copy.deepcopy(spot))

    if cache_spots:
        with open('spots_cached.pkl', 'wb') as wf:
            pickle.dump(all_spots, wf)


# Compute x, y, w, omega, and omega width for every spot
def compute_mean_spot(spot_list: list[TrackedSpot]) -> np.ndarray:
    # FIXME: we should review how we are computing the omega value for
    # the 3D spot, as well as the omega width.
    # Right now we are doing a width-weighted average for the omega value.
    # This seems to come up with a somewhat close answer as to what
    # `pull_spots()` produced before.
    # But, summed intensities of the spots might be a better weight for
    # the weighted averages than the width of the spot.
    # We are then doing the full omega ranges of the frames as the omega
    # width (we can probably do something a little better than that).
    omega_ranges = np.radians([omegas[s.frame_index] for s in spot_list])
    omega_values = [np.mean(x) for x in omega_ranges]
    widths = np.array([s.w for s in spot_list])
    max_width = widths.max()
    # FIXME: we need to take another look at `i, j` ordering
    coords = np.array([(s.j, s.i) for s in spot_list])

    # We are using a width-weighted omega as the average omega, currently
    width_weighted_omega = (omega_values * widths).sum() / (widths.sum())
    width_weighted_coords = (coords * widths[:, np.newaxis]).sum(
        axis=0
    ) / widths.sum()

    # We are using the full range of omegas
    omega_width = (omega_ranges[-1][1] - omega_ranges[0][0]) / 2
    return np.asarray(
        (*width_weighted_coords, max_width, width_weighted_omega, omega_width)
    )


# Combine spots on different frames that appear to belong to the
# same HKL, and perform weighted averages for computing their x, y
# and omega values.
spot_arrays = {}
for det_key, spots_dict in all_spots.items():
    array = np.empty((len(spots_dict), 5), dtype=float)
    for i, spot_list in enumerate(spots_dict.values()):
        array[i] = compute_mean_spot(spot_list)

    spot_arrays[det_key] = array


def in_range(x: np.ndarray, xrange: tuple[float, float]) -> np.ndarray:
    # Generic function to determine which `x` values are in the range `xrange`
    # Returns an array of booleans indicating which ones were in range
    return (xrange[0] <= x) & (x < xrange[1])


# Break up the spots into subpanels, and remap the coordinates
# to be within the subpanel's coordinates.
subpanel_spot_arrays = {}
for mono_det_key, array in spot_arrays.items():
    for det_key, panel in instr.detectors.items():
        if det_key[:3] != mono_det_key:
            # Not a part of this detector
            continue

        # Extract all spots that belong to this subpanel
        on_panel_rows = in_range(array[:, 0], panel.roi[1]) & in_range(
            array[:, 1], panel.roi[0]
        )
        if np.any(on_panel_rows):
            # Extract the spots on the subpanel
            subpanel_array = array[on_panel_rows]

            # Adjust the i, j coordinates for this subpanel
            subpanel_array[:, 0] -= panel.roi[1][0]
            subpanel_array[:, 1] -= panel.roi[0][0]
            subpanel_spot_arrays[det_key] = subpanel_array

# Compute tth, eta for the spot arrays
# We will use these to compare to the simulated spot
# results and match spots with HKLs.
angular_spot_arrays = {}
for det_key, array in subpanel_spot_arrays.items():
    panel = instr.detectors[det_key]

    # First convert to cartesian
    xys = panel.pixelToCart(array[:, [1, 0]])

    # Next convert to angles. Apply the distortion.
    ang_crds, _ = panel.cart_to_angles(
        xys,
        tvec_s=instr.tvec,
        apply_distortion=True,
    )

    # Map the angles to our eta period
    ang_crds[:, 1] = mapAngle(ang_crds[:, 1], eta_period)

    # Copy over all of the other stuff (width, omega, etc.)
    # and replace the x, y at the beginning with tth and eta
    new_array = array.copy()
    new_array[:, :2] = ang_crds

    angular_spot_arrays[det_key] = new_array

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

# Loop over detectors and grain IDs and try to locate their matching spots
det_hkl_assignments = {}
for det_key, sim_results in simulated_results.items():
    panel = instr.detectors[det_key]
    ang_spots = angular_spot_arrays[det_key]

    # 0 is tth, 1 is eta, and 3 is omega
    ang_spot_coords = ang_spots[:, [0, 1, 3]]
    raw_spot_coords = subpanel_spot_arrays[det_key][:, [0, 1, 3]]

    # Grab some simulated HKLs
    sim_all_hkls = sim_results[1]
    sim_all_xys = sim_results[3]

    # results[2] is angles, but these don't take into account things like
    # grain centroid shifts. It's more accurate to compute the angles from
    # the xys.

    detector_assigned_spots = []
    grain_hkl_assignments = det_hkl_assignments.setdefault(det_key, {})
    for grain_id, sim_xys in enumerate(sim_all_xys):
        sim_omegas = sim_results[2][grain_id][:, 2]
        sim_hkls = sim_all_hkls[grain_id]

        tvec_c = grain_params[grain_id][3:6]
        angles, _ = panel.cart_to_angles(sim_xys, tvec_c=tvec_c)

        # Fix eta period
        angles[:, 1] = mapAngle(angles[:, 1], eta_period)

        # Add the omegas
        angles = np.hstack((angles, sim_omegas[:, np.newaxis]))

        # Create the hkl assignments array
        hkl_assignments = np.full(len(sim_hkls), -1, dtype=int)
        skipped_spots = []
        assigned_spots = []
        for i, ang_crd in enumerate(angles):
            # Find the closest spot
            differences = abs(ang_crd - ang_spot_coords)
            distances = np.sqrt((differences**2).sum(axis=1))
            min_idx = distances.argmin()

            # Verify that the differences are within the tolerances
            if not np.all(differences[min_idx] < tolerances):
                # Skip this spot!!!
                skipped_spots.append(min_idx)
                continue

            hkl_assignments[i] = min_idx
            assigned_spots.append(min_idx)

        if skipped_spots:
            # FIXME: better handling here
            # This just means we identified some spots that were
            # not paired with HKLs. That might be okay, because
            # we might have not simulated all of the HKLs.
            print(
                f'For grain {grain_id} on detector {det_key}, did not '
                'pair these spots with HKLs:',
                skipped_spots,
            )

        assigned_spots = np.asarray(assigned_spots)
        assigned_indices_sorted, counts = np.unique(
            assigned_spots,
            return_counts=True,
        )
        if np.any(counts > 1):
            # FIXME: better handling here
            # This means two different HKLs were assigned to the same spot.
            # We'll definitely have to figure out what to do about this...
            print(
                f'WARNING!!! {grain_id} on detector {det_key}, '
                'some spots were assigned twice!',
                counts[counts > 1],
            )

        # Keep track of all spots assigned on this detector, so
        # we can figure out if any spots were assigned to multiple
        # grains.
        detector_assigned_spots.append(assigned_indices_sorted)

        cart_spot_coords = np.empty((len(assigned_spots), 3))
        meas_angs = np.empty((len(assigned_spots), 3))
        if assigned_spots.size != 0:
            cart_spot_coords[:, :2] = panel.pixelToCart(
                raw_spot_coords[assigned_spots][:, [1, 0]]
            )
            cart_spot_coords[:, 2] = raw_spot_coords[assigned_spots, 2]
            meas_angs = ang_spot_coords[assigned_spots]

        keep_hkls = hkl_assignments != -1

        # Store our assignments. `hkls[i]` is the HKL that corresponds
        # to both `sim_xys[i]`, `meas_xys[i]`, and `meas_angs[i]`
        grain_hkl_assignments[grain_id] = {
            'hkls': sim_hkls[keep_hkls],
            'sim_xys': sim_xys[keep_hkls],
            'meas_xys': cart_spot_coords,
            'assigned_spots': assigned_spots,
            'meas_angs': meas_angs,
        }

    # Check if any spots assigned to HKLs from one grain were also assigned
    # to HKLs on another grain
    detector_assigned_indices_sorted, counts = np.unique(
        np.hstack(detector_assigned_spots),
        return_counts=True,
    )
    if np.any(counts > 1):
        # FIXME: better handling here
        print(
            f'WARNING!!! On detector {det_key}, '
            'some spots were assigned to multiple grains!',
            counts[counts > 1],
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
max_omega_diff = 0

assigned_spots = {}
for grain_id, (ref_completeness, ref_grain_spots) in ref_spots_dict.items():
    for det_key, ref_det_spots in ref_grain_spots.items():
        # Grab the measured spots
        meas_spots = det_hkl_assignments[det_key][grain_id]

        # Keep track of which spots were assigned and which were not
        assigned_spots.setdefault(det_key, {})
        these_assigned_spots = np.zeros(len(meas_spots['hkls']), dtype=bool)
        assigned_spots[det_key][grain_id] = these_assigned_spots

        for ref_spot in ref_det_spots:
            hkl = ref_spot[2]
            ref_sum_int = ref_spot[3]
            ref_max_int = ref_spot[4]
            ref_meas_angs = ref_spot[6]
            ref_omega = ref_meas_angs[2]
            ref_meas_xy = ref_spot[7]

            matching_idx = -1
            for row in range(len(meas_spots['hkls'])):
                if np.array_equal(meas_spots['hkls'][row], hkl):
                    # Found a matching HKL!
                    matching_idx = row
                    break

            if matching_idx == -1:
                print(
                    f'Warning! Did not find a match on {det_key} for HKL: {hkl}'
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
            max_omega_diff = max(ome_diff, max_omega_diff)

            # Indicate this spot was assigned
            these_assigned_spots[matching_idx] = True
            num_matched += 1


max_distance = max(distances)
percent_found = num_matched / (num_matched + num_unmatched) * 100
print(
    'Percentage of spots that `pull_spots()` found that we also found:',
    f'{percent_found:.2f}%',
)

print(f'Mean distance (xy): {np.mean(distances):.4f}')
print(f'Max distance (xy): {max_distance:.4f}')
print(f'Max omega diff: {max_omega_diff:.4f}')
num_extra_hkls = 0
for det_key, det_assignments in assigned_spots.items():
    for grain_id, grain_assignments in det_assignments.items():
        if np.any(~grain_assignments):
            extra_hkls = det_hkl_assignments[det_key][grain_id]['hkls'][
                ~grain_assignments
            ]
            num_extra_hkls += len(extra_hkls)
            # print('Extra HKLs:', extra_hkls)

print(
    'Number of HKLs we found that `pull_spots()` did not find:', num_extra_hkls
)
