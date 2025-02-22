import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from hyperopt import fmin, tpe, hp, Trials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from qiskit import QuantumCircuit
from qiskit.circuit import Parameter
from qiskit.circuit.library import real_amplitudes, zz_feature_map
from qiskit_machine_learning.utils import algorithm_globals
from qiskit_machine_learning.neural_networks import SamplerQNN, EstimatorQNN
from qiskit_aer.primitives import EstimatorV2 as Estimator
from qiskit_machine_learning.connectors import TorchConnector


# Set seed for random generators
algorithm_globals.random_seed = 42
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 讀取數據
df = pd.read_csv("./testdata.csv")
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# 標準化特徵
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 資料集分割 80% 訓練, 10% 驗證, 10% 測試
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# 轉換為 PyTorch 張量
X_train, y_train = torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32)
X_val, y_val = torch.tensor(X_val, dtype=torch.float32), torch.tensor(y_val, dtype=torch.float32)
X_test, y_test = torch.tensor(X_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.float32)

train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(TensorDataset(X_val, y_val), batch_size=32, shuffle=False)

# statevector: 453
# auto: 450
# estimator with all default setting: 216
# estimator = Estimator(options={'backend_options': {"device": "GPU", "cuStateVec_enable": True}})
estimator = Estimator()

def create_qnn():
    feature_map = zz_feature_map(2)
    ansatz = real_amplitudes(2, reps=1)
    qc = QuantumCircuit(2)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    # REMEMBER TO SET input_gradients=True FOR ENABLING HYBRID GRADIENT BACKPROP
    qnn = EstimatorQNN(
        circuit=qc,
        input_params=feature_map.parameters,
        weight_params=ansatz.parameters,
        input_gradients=True,
        estimator=estimator,
    )
    return qnn

class DiabetesMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, ffn_layer_num, dropout, qnn):
        super(DiabetesMLP, self).__init__()
        layers = []

        for i in range(ffn_layer_num):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            input_dim = hidden_dim
            hidden_dim = max(hidden_dim // 2, 2)
        layers.append(nn.Linear(input_dim, 2))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        # QNN layer
        # self.qnn = TorchConnector(qnn)
        # layers.append(self.qnn)

        layers.append(nn.Linear(2, 1))
        layers.append(nn.Sigmoid())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

space = {
    "hidden_dim": hp.quniform("hidden_dim", 32, 128, 8),  
    "ffn_layer_num": hp.quniform("ffn_layer_num", 2, 5, 1),
    "dropout": hp.uniform("dropout", 0.0, 0.5),
    "lr": hp.loguniform("lr", np.log(1e-4), np.log(1e-2)),
}

def train_model(params):
    hidden_dim = int(params["hidden_dim"])
    ffn_layer_num = int(params["ffn_layer_num"])
    dropout = params["dropout"]
    lr = params["lr"]

    qnn = create_qnn()
    model = DiabetesMLP(input_dim=X_train.shape[1], hidden_dim=hidden_dim,
                        ffn_layer_num=ffn_layer_num, dropout=dropout,
                        qnn=qnn).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    best_val_loss = float("inf")
    patience, patience_limit = 0, 5

    for epoch in range(20):
        model.train()
        total_loss = 0
        val_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if epoch == 19:
            model.eval()
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
                    outputs = model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience = 0
            else:
                patience += 1
                if patience >= patience_limit:
                    break
    
    val_loss /= len(val_loader)
    return val_loss

trials = Trials()
best_params = fmin(
    fn=train_model,
    space=space,
    algo=tpe.suggest,
    max_evals=20,
    trials=trials,
)

print("Best Hyperparameters:", best_params)
print(trials.results)
print("Best result:", trials.best_trial)
