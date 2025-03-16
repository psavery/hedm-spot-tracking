import copy
from pathlib import Path

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
    ops = [('rectangle', panel.roi), ]

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

# FIXME: we are caching results for testing purposes...
if Path('spots_cached.pkl').exists():
    # Just load the cached results...
    import pickle
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

    # FIXME: we are caching results for testing purposes...
    import pickle
    with open('spots_cached.pkl', 'wb') as wf:
        pickle.dump(all_spots, wf)


# Compute x, y, w, omega, and omega width for every spot
def compute_mean_spot(spot_list: list[TrackedSpot]) -> np.ndarray:
    # FIXME: should we come up with a better way to compute mean omega?
    # FIXME: should we mean the widths?
    omega_ranges = np.radians([omegas[x.frame_index] for x in spot_list])
    omega_value = (omega_ranges[0][0] + omega_ranges[-1][1]) / 2
    omega_width = (omega_ranges[-1][1] - omega_ranges[0][0]) / 2
    means = np.array([(x.x, x.y, x.w) for x in spot_list]).mean(axis=0)
    return np.hstack((means, omega_value, omega_width))


# Create spot arrays for each detector
spot_arrays = {}
for det_key, spots_dict in all_spots.items():
    array = np.empty((len(spots_dict), 5), dtype=float)
    for i, spot_list in enumerate(spots_dict.values()):
        array[i] = compute_mean_spot(spot_list)

    spot_arrays[det_key] = array


def in_range(x: float, xrange: tuple[float, float]) -> bool:
    return (xrange[0] <= x) & (x < xrange[1])


# Break up the arrays into composite detectors
subpanel_spot_arrays = {}
for mono_det_key, array in spot_arrays.items():
    for det_key, panel in instr.detectors.items():
        if det_key[:3] != mono_det_key:
            # Not a part of this detector
            continue

        # Extract all spots that belong to this subpanel
        on_panel_rows = (
            in_range(array[:, 0], panel.roi[0]) &
            in_range(array[:, 1], panel.roi[1])
        )
        if np.any(on_panel_rows):
            # Extract the subpanel
            subpanel_array = array[on_panel_rows]

            # Adjust the i, j coordinates for this subpanel
            subpanel_array[:, 0] -= panel.roi[0][0]
            subpanel_array[:, 1] -= panel.roi[1][0]
            subpanel_spot_arrays[det_key] = subpanel_array

# Compute tth, eta for the spot arrays
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
tth_tol = np.radians(0.5)
eta_tol = np.radians(0.5)
ome_tol = np.radians(0.5)
tolerances = np.array([tth_tol, eta_tol, ome_tol])

# Loop over detectors and grain IDs and try to locate their matching spots
for det_key, results in simulated_results.items():
    ang_spots = angular_spot_arrays[det_key]
    # 0 is tth, 1 is eta, and 3 is omega
    ang_spot_coords = ang_spots[:, [0, 1, 3]]
    all_hkls = results[1]
    all_angs = results[2]
    for grain_id, angs in enumerate(all_angs):
        hkls = all_hkls[grain_id]

        # Create the hkl assignments array
        hkl_assignments = np.full(len(hkls), -1, dtype=int)
        for i, ang_crd in enumerate(angs):
            # Find the closest spot
            differences = abs(ang_crd - ang_spot_coords)
            distances = np.sqrt((differences**2).sum(axis=1))
            min_idx = distances.argmin()

            # Verify that the differences are within the tolerances
            if np.all(differences[min_idx] > tolerances):
                # Skip this spot!!!
                continue

            hkl_assignments[i] = min_idx

        # FIXME: Check if there were any HKLs assigned to the same spot
