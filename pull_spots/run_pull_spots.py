from pathlib import Path
import pickle

import h5py
import numpy as np
import yaml

from hexrd import imageseries
from hexrd.imageseries.omega import OmegaImageSeries
from hexrd.imageseries.process import ProcessedImageSeries
from hexrd.instrument import HEDMInstrument
from hexrd.material.material import load_materials_hdf5

script_path = Path(__file__).resolve()
parent_dir_path = script_path.parent.parent

instr_file = parent_dir_path / 'dual_dexelas_composite.yml'
image_files = {
    'ff1': parent_dir_path / 'mruby-0129_000004_ff1_000012-cachefile.npz',
    'ff2': parent_dir_path / 'mruby-0129_000004_ff2_000012-cachefile.npz',
}
materials_file = parent_dir_path / 'ruby.h5'
grain_params_file = parent_dir_path / 'grain_params.npy'

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
    ims_dict[det_key] = OmegaImageSeries(ProcessedImageSeries(ims, ops))

# Load the material
with h5py.File(materials_file, 'r') as rf:
    material = load_materials_hdf5(rf)['ruby']

plane_data = material.planeData

# Load the known grain parameters
grain_params = np.load(grain_params_file)

outputs = {}
for i, params in enumerate(grain_params):
    kwargs = {
        'plane_data': plane_data,
        'grain_params': params,
        'tth_tol': 0.25,
        'eta_tol': 1.00,
        'ome_tol': 1.50,
        'imgser_dict': ims_dict,
        'npdiv': 4,
        'threshold': 25.0,
        'eta_ranges': eta_ranges,
        'ome_period': omega_period,
        'dirname': None,
        'filename': None,
        'return_spot_list': False,
        'quiet': True,
        'check_only': False,
        'interp': 'nearest',
    }
    outputs[i] = instr.pull_spots(**kwargs)

with open('pull_spots_output.pkl', 'wb') as wf:
    pickle.dump(outputs, wf)
