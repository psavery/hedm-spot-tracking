import copy

import h5py
import numpy as np
import yaml

from hexrd import imageseries
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument import HEDMInstrument
from hexrd.material.material import load_materials_hdf5

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
omega_period = np.radians(omegas[0][0], omegas[-1][1])

# Just assume the whole eta and omega ranges are used
eta_ranges = [(-np.pi, np.pi)]
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
for frame_index in range(num_images):
    for det_key, ims in raw_ims_dict.items():
        # First, find spots
        img = ims[frame_index]
        spots = finder.find_spots(img)

        # Next, track spots
        tracker = spot_trackers[det_key]
        tracker.track_spots(spots)

        # Now update our tracked list
        spot_dict = all_spots[det_key]
        for spot_id, spot in tracker.current_spots.items():
            spot_list = spot_dict.setdefault(spot_id, [])
            spot_list.append(copy.deepcopy(spot))

# Now simulate the spots
simulated_results = instr.simulate_rotation_series(
    plane_data,
    grain_params,
    eta_ranges,
    omega_ranges,
    omega_period,
)

# Loop over detectors and grain IDs and try to locate their matching spots
for det_key, results in simulated_results.items():
    all_hkls = results[1]
    all_angs = results[2]
    all_xys = results[3]
    for grain_id, xys in enumerate(all_xys):
        hkls = all_hkls[grain_id]

        # FIXME: identify matching spots
