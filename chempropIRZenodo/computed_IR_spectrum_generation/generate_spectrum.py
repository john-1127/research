import sys, os, argparse
sys.path.append('/global/homes/y/yanfeig/bin')

import pandas as pd
import numpy as np
import json
from joblib import Parallel, delayed

from g16_log import XtbLog, G16Log
from create_spectrum import set_x_spacing, fixed_var_spectrum, boltzmann, norm_integration, pytorch_model_spectrum, load_pytorch_model


model = load_pytorch_model('model19.pt')
parser = argparse.ArgumentParser(description='generating spectrum')
parser.add_argument('--min', required=False, default=400,
                    help='minimum range for the reported spectrum')
parser.add_argument('--max', required=False, default=4000,
                    help='maximum range for the reported spectrum')
parser.add_argument('--wn_spacing', required=False, default=2,
                    help='spacing')
parser.add_argument('--norm_min', required=False, default=500,
                    help='norm range min')
parser.add_argument('--norm_max', required=False, default=1500,
                    help='norm range max')
parser.add_argument('-s', '--smiles_df', required=True,
                    help='csv for smiles')
parser.add_argument('-t', '--type', default='xtb', choices=['xtb', 'g16'])
parser.add_argument('target',
                    help='target dictionary')
args = parser.parse_args()

def create_spectrum(freq_logs, args, input_type='xtb'):
    norm_min = args.norm_min
    norm_max = args.norm_max
    xs = set_x_spacing(args.max, args.min, args.wn_spacing)
    gs = []
    spectra = []
    for freq_log in freq_logs:
        if input_type == 'xtb':
            try:
                log = XtbLog(freq_log)
            except:
                continue
        elif input_type == 'g16':
            log = G16Log(freq_log)

        if not log.termination:
            continue

        if min(log.wavenum) < 0:
            continue

        if input_type == 'g16':
            log.wavenum = log.har_wavenumbers + log.an_wavenumbers + log.over_wavenumbers + log.com_wavenumbers
            log.ir_intensities = log.har_intensities + log.an_intensities + log.over_intensities + log.com_intensities

        gs.append(log.G)
        spectra.append(pytorch_model_spectrum(log.wavenum, log.ir_intensities, xs,model))

    if spectra:
        dist = np.array(boltzmann(gs)).reshape((-1, 1))
        spectra_sum = np.sum(np.array(spectra)*dist, axis=0)
        spectra_final = norm_integration(spectra_sum, xs, norm_min, norm_max).tolist()

        spectra_final = {'frequencies': xs.tolist(), 'intensities': spectra_final}

        return spectra_final
    else:
        print(freq_logs)
        return None


freq_dicts = {}
freqs = [x for x in os.listdir(args.target) if '_freq.log' in x]
for freq in freqs:
    m = freq.split('_')[0]
    freq = os.path.join(dir, freq)
    try:
        freq_dicts[m].append(freq)
    except KeyError as e:
        freq_dicts[m] = [freq]

spectrums = Parallel(n_jobs=-1, verbose=10)(delayed(create_spectrum)(freq_dicts[freq], args) for freq in freq_dicts)
spectrums_all = []
smiles_df = pd.read_csv(args.smiles_df, index_col=0).set_index('CID')
for freq, spectrum in zip(freq_dicts, spectrums):
    if spectrum:
        smiles = smiles_df.loc[freq]['SMILES']
        spectrum['smiles'] = smiles
        spectrums_all.append(spectrum)
    else:
        continue

with open('spectrum_collected.json', 'w') as output:
    json.dump(spectrums_all, output)

