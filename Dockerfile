# For server196
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive


# Define the IBM Quantum API token as an environment variable
ARG IBM_QUANTUM_TOKEN
ENV IBM_QUANTUM_TOKEN=${IBM_QUANTUM_TOKEN}

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
libglib2.0-0 libxext6 libsm6 libxrender1 build-essential \
git mercurial subversion libbz2-dev libz-dev libpng-dev graphviz \
&& apt-get clean && rm -rf /var/lib/apt/lists/*

RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
/bin/bash ~/miniconda.sh -b -p /opt/conda && \
rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

COPY environment.yml /project/environment.yml

RUN conda install pip
RUN conda install -c rdkit nox
RUN conda install cairo
RUN conda install -c conda-forge jupyterlab notebook ipykernel pandas matplotlib
RUN pip install git+https://github.com/bp-kelley/descriptastorus
RUN pip install seaborn
RUN pip install 'qiskit[visualization]' qiskit-ibm-runtime qiskit-aer-gpu qiskit-machine-learning pylatexenc
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
RUN conda env update -n base --file /project/environment.yml
RUN python -c "\
from qiskit_ibm_runtime import QiskitRuntimeService; \
QiskitRuntimeService.save_account(\
    token='${IBM_QUANTUM_TOKEN}', \
    channel='ibm_quantum'\
)"
# For import chempropIR
ENV PYTHONPATH=/project/chempropIRZenodo/chempropIR:$PYTHONPATH

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]