import sys, os, argparse
sys.path.append('/global/homes/y/yanfeig/bin')

import pandas as pd
from tqdm import tqdm
import json
from multiprocessing import Process

from g16_log import XtbLog, G16Log
from create_spectrum import set_x_spacing

parser = argparse.ArgumentParser(description='generating spectrum')
parser.add_argument('--min', required=False, default=400,
                    help='minimum range for the reported spectrum')
parser.add_argument('--max', required=False, default=4000,
                    help='maximum range for the reported spectrum')
parser.add_argument('--wn_spacing', required=False, default=2,
                    help='spacing')
parser.add_argument('--fixed_var', required=False, default=50,
                    help='fixed_var for gaussian distribution')
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

def create_spectrum(freq_logs, input_type):
    spectra = None
    Gmin = 100000
    for freq_log in freq_logs:
        if input_type == 'xtb':
            log = XtbLog(freq_log)
        elif input_type == 'g16':
            log = G16Log(freq_log)

        if not log.termination:
            continue

        if input_type == 'g16':
            log.wavenum = log.har_wavenumbers + log.an_wavenumbers + log.over_wavenumbers + log.com_wavenumbers
            log.ir_intensities = log.har_intensities + log.an_intensities + log.over_intensities + log.com_intensities

        if min(log.wavenum) < 0:
            continue

        if log.G < Gmin:
            spectra = {'peaks': log.wavenum, 'intensity': log.ir_intensities}
            Gmin == log.G

    return spectra

sub_dirs = [os.path.join(args.target, x) for x in os.listdir(args.target) if 'split_' in x and os.path.isdir(os.path.join(args.target, x))]


def search_dir(dir):
    freqs = [x for x in os.listdir(dir) if '_freq.log' in x]
    smiles_df = pd.read_csv(args.smiles_df, index_col=0).set_index('CID')
    
    freq_dicts = {}
    for freq in freqs:
        m = freq.split('_')[0]
        freq = os.path.join(dir, freq)
        try:
            freq_dicts[m].append(freq)
        except KeyError as e:
            freq_dicts[m] = [freq]
    
    xs = set_x_spacing(args.max, args.min, args.wn_spacing)
    
    spectrum_all = []
    for freq in tqdm(list(freq_dicts.keys()), total=len(freq_dicts)):
        try:
            spectrum = create_spectrum(freq_dicts[freq], args.type, xs, args.norm_min, args.norm_max, args.fixed_var)
            if not spectrum:
                continue

            smiles = smiles_df.loc[int(freq)]['SMILES']
            spectrum['smiles'] = smiles
            spectrum_all.append(spectrum)
        except:
            continue
   
    with open('{}_peaks.json'.format(dir), 'w') as output:
        json.dump(spectrum_all, output, indent=4)

procs = []
for dir in sub_dirs:
    proc = Process(target=search_dir, args=(dir,))
    procs.append(proc)
    proc.start()

for proc in procs:
    proc.join()








