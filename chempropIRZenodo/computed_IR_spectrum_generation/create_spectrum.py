#!/usr/bin/python3

import numpy as np

def boltzmann(energies):
    q=list()
    e_offset=min(energies)
    for i in energies:
        q.append(np.exp((i-e_offset)/(8.31446261815324/1000*298.15)))
    q_sum=sum(q)
    dist=list()
    for i in q:
        prob=i/q_sum
        dist.append(prob)
    return dist


def set_x_spacing(plot_range_max, plot_range_min, wave_number_spacing):
    remainder=(plot_range_max-plot_range_min)%wave_number_spacing
    if remainder==0:
        npoints=int((plot_range_max-plot_range_min)//wave_number_spacing+1)
    if remainder!=0:
        npoints=int((plot_range_max-plot_range_min)//wave_number_spacing+2)
    xs=np.linspace(plot_range_min,plot_range_max,npoints)
    return xs


def fixed_var_spectrum(frequencies,intensities,xs,fixed_var):
    ys=np.empty((0,len(xs)))
    for i in range(len(frequencies)):
        y=intensities[i]/(2*np.pi*fixed_var)**0.5*np.exp(-(xs-frequencies[i])**2/(2*fixed_var))
        if frequencies[i]==0.00:
            y=y*0.0
        y=np.expand_dims(y,axis=0)
        ys=np.append(ys,y,axis=0)
    spectrum_data=ys.sum(axis=0)
    return spectrum_data


def norm_integration(spectra_sum, xs, norm_range_min, norm_range_max):
    norm_factor=sum([j for i,j in enumerate(spectra_sum) if xs[i]>=norm_range_min and xs[i]<norm_range_max])
    norm_spectrum=spectra_sum/norm_factor
    return norm_spectrum

def pytorch_model_spectrum(frequencies,intensities,xs,model):
    peaks=torch.zeros([len(frequencies),1801])
    for e,i in enumerate(frequencies):
        peaks[e,:int(i//2-199)]=1
        if i < 4000: peaks[e,int(i//2-199)]=i%2/2
    weights=torch.tensor(intensities)
    peaks=model(peaks)
    peaks=torch.exp(peaks)
    weights=torch.unsqueeze(weights,dim=1)
    peaks=torch.mul(peaks,weights)
    output=torch.sum(peaks,dim=0)
    return output.data.numpy()

def load_pytorch_model(model_path):
    state=torch.load(model_path,map_location=torch.device('cpu'))
    w0=state['state_dict']['peaknn.0.weight']
    b0=state['state_dict']['peaknn.0.bias']
    w2=state['state_dict']['peaknn.2.weight']
    b2=state['state_dict']['peaknn.2.bias']
    torch.set_default_dtype(torch.double)
    L0=torch.nn.Linear(1801,2200)
    L0.weight=torch.nn.Parameter(w0)
    L0.bias=torch.nn.Parameter(b0)
    L1=torch.nn.ReLU()
    L2=torch.nn.Linear(2200,1801)
    L2.weight=torch.nn.Parameter(w2)
    L2.bias=torch.nn.Parameter(b2)
    model=torch.nn.Sequential(L0,L1,L2)
    for param in model.parameters():
        param.requires_grad=False
    return model