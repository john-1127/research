# Computaional workflow for IR spectrum
This directory contains code for IR spectrum calculation through the semi-empirical method.

## Usage

### Generating conformers
Conformers are generated and minimized through MMFF94s force field within RDKit

```
python genConf.py -ismiles <input_smiles_file>.csv -osdf <output folder>
```

`<input_smiles_file>.csv` is the input file storing interested SMILES strings. The `.csv` file must contain two columns `CID` and `SMILES`.
For example:
```
,CID,SMILES
0,00001,CCC
1,00002,CCCC
```
Generated conformers will be stored in `.sdf` files in `<output folder>`.

### Structure optimization and  harmonic frequency calculation
The workflow optimizes structures for generated conformers under GFN2-XTB level of theory and then calculates the harmonic frequency. 
```
python xtb.py <output folder>
```
where the `<output folder>` is the same as that in the [Generating conformers](#Generating conformers).

PS: the external `xtb` command is called through the `subprocess.call` in the `xtb.py`. You may need to configure `xtb.py` to use the 
specific path to `xtb` installed on your machine.

### Generate spectrum
The workflow extracts harmonic frequency from output files of `XTB` calculations, and then generates corresponding IR spectrum by spreading 
each harmonic peak. Peak spreading is carried out according to peak shapes fitted to experiment data, as stored in file `model.pt`.
For multiple conformers for each given SMILES string, the workflow conducts a Boltzmann weighting over all available conformers
by using `GFX-XTB` free energies.

```
python generate_spectrum.py <ouptut folder>
```

The `<output folder>` is the same as the one in [Generating conformers](#Generating conformers) and [Structure optimization and  harmonic frequency calculation](#Structure optimization and  harmonic frequency calculation).
The generated spectrum will be stored in `spectrum_collected.json` for all molecules.