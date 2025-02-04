from rdkit import Chem
import os, sys
import re
import pandas as pd
import numpy as np

periodictable = ["", "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar",
                 "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br",
                 "Kr", "Rb", "Sr", "Y", "Zr",
                 "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La",
                 "Ce", "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W",
                 "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl",
                 "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf",
                 "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Uub", "Uut", "Uuq",
                 "Uup", "Uuh", "Uus", "Uuo"]


def elementID(massno):
    if massno < len(periodictable):
        return periodictable[massno]
    else:
        return "XX"


class G16Log:
    def __init__(self, file):
        # default values for thermochemical calculations
        if '.log' not in file:
            raise TypeError('A g16 .log file must be provided')

        self.file = file
        self.name = os.path.basename(file)

        self.GetTermination()
        if not self.termination:
            self.GetError()
        else:
            self.GetCoords()
            self.GetNPA()
            self.GetCPU()

    def GetTermination(self):
        with open(self.file) as fh:
            for line in (fh):
                if line.find("Normal termination") > -1:
                    self.termination = True
                    return True
            self.termination = False

    def GetError(self):
        with open(self.file) as fh:
            for line in (fh):
                if line.find("Error termination") > -1:
                    self.error = line
                    return True
            self.error = None

    def GetCPU(self):
        with open(self.file) as fh:
            for line in (fh):
                if line.find("Job cpu time") > -1:
                    days = int(line.split()[3])
                    hours = int(line.split()[5])
                    mins = int(line.split()[7])
                    secs = float(line.split()[9])

                    self.CPU = [days, hours, mins, secs]
                    break

    def GetCoords(self):
        with open(self.file) as fh:
            starting = False
            found_coord = False
            for line in (fh):
                if line.find('orientation') > -1:
                    starting = True
                    self.AtomsNum = []
                    self.AtomsType = []
                    self.Coords = []
                    sep = 0
                    found_coord = False
                if starting:
                    m = re.search('(\d+)\s+(\d+)\s+(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)',
                                  line)
                    if not m: continue
                    self.AtomsNum.append(int(m.group(2)))
                    self.AtomsType.append(elementID(int(m.group(2))))
                    self.Coords.append([float(m.group(4)), float(m.group(5)), float(m.group(6))])
                    found_coord = True
                if found_coord and line.find('-----------') > -1:
                    starting = False
                    found_coord = False

    def GetNPA(self):
        with open(self.file) as fh:
            txt = fh.readlines()

        txt = [x.strip() for x in txt]
        # charge and multiplicity
        for i, line in enumerate(txt):
            if line.find('Multiplicity') == -1:
                continue

            m = re.findall(r'-?\d', line)

            if m:
                txt = txt[i:]
                break
        if m:
            if int(m[0]) != 0:
                only_charge = True
            else:
                only_charge = False

        else:
            return None

        # NPA charge
        NPA_Charge = np.zeros([len(self.AtomsNum), 3])
        for i, line in enumerate(txt):
            m = re.search('Atom\s+No\s+Charge\s+Core\s+Valence\s+Rydberg\s+Total', line)
            if m:
                txt = txt[i + 2:]
                break

        for i, line in enumerate(txt):
            if re.search('=====', line):
                txt = txt[i + 1:]
                break
            m = re.search('(\S+)\s*(\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)',
                          line)
            NPA_Charge[i, :] = [float(m[3]), float(m[5]), float(m[6])]
        self.NPA_Charge = NPA_Charge

        if only_charge:
            return

        # valence electron configuration
        for i, line in enumerate(txt):
            if line.find('Natural Electron Configuration') > -1:
                txt = txt[i + 2:]
                break

        electron_configuration = []
        for i, line in enumerate(txt):
            m = [float(x[1]) for x in re.findall(r'(\d+\S+)\(\s?(-?\d+\.\d+)\)', line)]
            if not m:
                txt = txt[i + 1:]
                break
            else:
                electron_configuration.append(m)

        nc_np = np.zeros([len(self.AtomsNum), 5])
        for i, itr in enumerate(electron_configuration):
            nc_np[i, :len(itr)] = itr

        self.electron_configuration = nc_np

        # bond index
        for i, line in enumerate(txt):
            if line.find('Wiberg bond index matrix in the NAO basis') > -1:
                txt = txt[i + 2:]
                break

        keep_going = True
        bond_index_matrix = np.zeros([len(self.AtomsNum), len(self.AtomsNum)], dtype='float32')
        while keep_going:
            for i, line in enumerate(txt):
                m = re.findall('\s+(\d+)', line)
                if m:
                    txt = txt[i + 2:]
                    start, end = int(m[0]) - 1, int(m[-1])

                    if end == len(self.AtomsNum):
                        keep_going = False

                    break

            for i, line in enumerate(txt):
                m = re.findall(r'\d+\.\d+', line)
                if not m:
                    txt = txt[i + 1:]
                    break
                else:
                    bond_index_matrix[i, start:end] = [float(x) for x in m]

        self.bond_index_matrix = bond_index_matrix

        # occupancy of lewis structure
        for i, line in enumerate(txt):
            if line.find('(Occupancy)   Bond orbital / Coefficients / Hybrids') > -1:
                txt = txt[i + 2:]
                break

        txt_generator = (x for x in txt)
        keep_going = True
        non_lewis = False
        lone_pairs = np.zeros([len(self.AtomsNum), 4], dtype='float32')

        bond_lewis = np.zeros([len(self.AtomsNum), len(self.AtomsNum), 3], dtype='float32')
        bond_lewis_contribution = np.zeros([len(self.AtomsNum), len(self.AtomsNum), 3, 2], dtype='float32')
        bond_non_lewis = np.zeros([len(self.AtomsNum), len(self.AtomsNum), 3], dtype='float32')
        bond_non_lewis_contribution = np.zeros([len(self.AtomsNum), len(self.AtomsNum), 3, 2], dtype='float32')

        while keep_going:
            line = next(txt_generator)
            if line.find('non-Lewis') > -1:
                non_lewis = True

            m = re.search('\((\d+\.\d+)\)\s+LP\s+\(\s+(\d+)\)\s+\S+\s+(\d+)', line)
            if m:
                try:
                    occu, i, atom_num = float(m[1]), int(m[2]), int(m[3])
                except:
                    print(self.file)
                    print(line)
                lone_pairs[atom_num - 1, i - 1] = occu
                continue

            m = re.search('\((\d+\.\d+)\)\s+BD[\*\s]+\(\s+(\d+)\)\s+\S+\s+(\d+)-\s+\S+\s+(\d+)', line)
            if m:
                occu, i, start, end = float(m[1]), int(m[2]) - 1, int(m[3]) - 1, int(m[4]) - 1
                line = next(txt_generator)
                p1 = float(re.search('(\d+\.\d+)%', line)[1]) / 100

                while True:
                    line = next(txt_generator)
                    m = re.search('(\d+\.\d+)%', line)
                    if m:
                        p2 = float(m[1]) / 100
                        break

                if not non_lewis:
                    bond_lewis[start, end, i] = occu
                    bond_lewis[end, start, i] = occu
                    bond_lewis_contribution[start, end, i, :] = [p1, p2]
                    bond_lewis_contribution[end, start, i, :] = [p2, p1]

                else:
                    bond_non_lewis[start, end, i] = occu
                    bond_non_lewis[end, start, i] = occu
                    bond_non_lewis_contribution[start, end, i, :] = [p1, p2]
                    bond_non_lewis_contribution[end, start, i, :] = [p2, p1]

            if not line.strip():
                keep_going = False

        self.lone_pairs = lone_pairs
        self.bond_lewis = bond_lewis
        self.bond_non_lewis = bond_non_lewis
        self.bond_lewis_contribution = bond_lewis_contribution
        self.bond_non_lewis_contribution = bond_non_lewis_contribution


class XtbLog:
    def __init__(self, file):
        # default values for thermochemical calculations
        if '.log' not in file:
            raise TypeError('A xtb .log file must be provided')

        self.file = file
        self.name = os.path.basename(file)

        self.GetTermination()
        if not self.termination:
            self.GetError()
        else:
            self.GetFreq()
            self.GetE()

    def GetTermination(self):
        with open(self.file) as fh:
            for line in (fh):
                if line.find("normal termination") > -1:
                    self.termination = True
                    return True
            self.termination = False

    def GetFreq(self):
        with open(self.file) as fh:
            txt = fh.readlines()

        txt = [x.strip() for x in txt]
        for i, line in enumerate(txt):
            if line.find('Frequency Printout') > -1:
                txt = txt[i + 3:]
                break

        waveNums = []
        for i, line in enumerate(txt):
            if line.find('reduced masses') > -1:
                txt = txt[i + 1:]
                break
            m = re.findall('\s+(-?\d+\.\d+)', line)
            if m:
                for match in m:
                    waveNums.append(float(match.strip()))

        for i, line in enumerate(txt):
            if line.find('IR intensities') > -1:
                txt = txt[i + 1:]
                break

        intensities = []
        for i, line in enumerate(txt):
            if line.find('Raman intensities') > -1:
                txt = txt[i + 1:]
                break
            m = re.findall('\d+:\s+(\d+\.\d+)', line)
            if m:
                for match in m:
                    intensities.append(float(match))

        waveNums,intensities = list(zip(*[(w, i) for w, i in zip(waveNums, intensities) if w != 0]))

        if waveNums and intensities and len(waveNums) == len(intensities):
            self.wavenum = waveNums
            self.ir_intensities = intensities

    def GetE(self):
        with open(self.file) as fh:
            txt = fh.readlines()

        txt = [x.strip() for x in txt]
        for i, line in enumerate(txt):
            m = re.search('TOTAL ENERGY\s+(-?\d+\.\d+)', line)
            if m:
                self.E = m[1]
                continue
            m = re.search('TOTAL FREE ENERGY\s+(-?\d+\.\d+)', line)
            if m:
                self.G = float(m[1])

        
