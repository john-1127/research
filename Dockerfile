# For server196
FROM nvidia/cuda:12.0.1-cudnn8-devel-ubuntu22.04

RUN apt-get update --fix-missing && apt-get install -y wget bzip2 ca-certificates \
    libglib2.0-0 libxext6 libsm6 libxrender1 git \
    curl software-properties-common fontconfig unzip ripgrep fd-find build-essential gcc make clang file \
    mercurial subversion libbz2-dev libz-dev libpng-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# install Miniconda
RUN echo 'export PATH=/opt/conda/bin:$PATH' > /etc/profile.d/conda.sh && \
    wget --quiet https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh

ENV PATH /opt/conda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda-8.0/lib64:/usr/local/cuda-8.0/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# copy Conda environment
COPY environment.yml /project/environment.yml

# install conda dependencies
RUN conda install pip && \
    conda install -c rdkit nox && \
    conda install cairo && \
    conda env update -n base --file /project/environment.yml && \
    pip install git+https://github.com/bp-kelley/descriptastorus

# install pytorch
# RUN pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
RUN pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
# env chempropIR
ENV PYTHONPATH=/project/chempropIRZenodo/chempropIR:$PYTHONPATH

# 安裝 Homebrew
RUN git clone https://github.com/Homebrew/brew /home/linuxbrew/.linuxbrew && \
    echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >> /root/.profile && \
    echo 'eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)' >> /etc/profile && \
    eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv) && \
    brew update

# nvim install & homebrew
RUN eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv) && \
    brew install neovim && \
    ln -s /home/linuxbrew/.linuxbrew/bin/nvim /usr/local/bin/nvim

RUN eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv) && \
    nvim --version

# install Nerd Fonts
RUN mkdir -p /usr/share/fonts/nerdfonts && \
    wget -qO /usr/share/fonts/nerdfonts/Ubuntu.zip https://github.com/ryanoasis/nerd-fonts/releases/download/v3.3.0/Ubuntu.zip && \
    unzip /usr/share/fonts/nerdfonts/Ubuntu.zip -d /usr/share/fonts/nerdfonts && \
    fc-cache -fv && \
    rm /usr/share/fonts/nerdfonts/Ubuntu.zip

ENV TZ=Asia/Taipei
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /project

# LazyVim
RUN git clone https://github.com/LazyVim/starter ~/.config/nvim

# Homebrew
RUN eval $(/home/linuxbrew/.linuxbrew/bin/brew shellenv)

