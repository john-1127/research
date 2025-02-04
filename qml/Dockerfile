# Use the official Python image (based on Ubuntu)
FROM python:3.12-slim

# Set environment variables to avoid interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive


# Define the IBM Quantum API token as an environment variable
ARG IBM_QUANTUM_TOKEN
ENV IBM_QUANTUM_TOKEN=${IBM_QUANTUM_TOKEN}

# Update and install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install the required Python packages
RUN pip install 'qiskit[visualization]' jupyterlab notebook ipykernel pandas matplotlib pylatexenc qiskit-ibm-runtime qiskit-aer-gpu qiskit-machine-learning

# Create a working directory
WORKDIR /app

# Save the IBM Quantum account using the environment variable token
RUN python -c "\
from qiskit_ibm_runtime import QiskitRuntimeService; \
QiskitRuntimeService.save_account(\
    token='${IBM_QUANTUM_TOKEN}', \
    channel='ibm_quantum'\
)"


RUN apt-get update && apt-get install -y graphviz
RUN pip install seaborn

# Expose the default Jupyter Notebook port
EXPOSE 8888

# Set the default command to start Jupyter Notebook
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
