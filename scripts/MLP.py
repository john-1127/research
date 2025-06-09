import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd


def sid(model_spectra: torch.tensor, target_spectra: torch.tensor, threshold: float = 1e-8, eps: float = 1e-8, torch_device: str = 'cpu') -> torch.tensor:
    # normalize the model spectra before comparison
    nan_mask=torch.isnan(target_spectra)+torch.isnan(model_spectra)
    nan_mask=nan_mask.to(device=torch_device)
    zero_sub=torch.zeros_like(target_spectra,device=torch_device)
    model_spectra = model_spectra.to(torch_device)
    model_spectra[model_spectra < threshold] = threshold
    sum_model_spectra = torch.sum(torch.where(nan_mask,zero_sub,model_spectra),axis=1)
    sum_model_spectra = torch.unsqueeze(sum_model_spectra,axis=1)
    model_spectra = torch.div(model_spectra,sum_model_spectra)
    # calculate loss value
    if not isinstance(target_spectra,torch.Tensor):
        target_spectra = torch.tensor(target_spectra)
    target_spectra = target_spectra.to(torch_device)
    loss = torch.ones_like(target_spectra)
    loss = loss.to(torch_device)
    target_spectra[nan_mask]=1
    model_spectra[nan_mask]=1
    loss = torch.mul(torch.log(torch.div(model_spectra,target_spectra)),model_spectra) \
        + torch.mul(torch.log(torch.div(target_spectra,model_spectra)),target_spectra)
    loss[nan_mask]=0
    loss = torch.sum(loss,axis=1)
    return loss


class QFFDataset(Dataset):
    def __init__(self, qnn_path, ffnn_path, target_path):
        self.qnn_df = pd.read_csv(qnn_path)
        self.ffnn_df = pd.read_csv(ffnn_path)
        self.target_df = pd.read_csv(target_path)

        assert all(self.qnn_df["smiles"] == self.ffnn_df["smiles"])
        assert all(self.qnn_df["smiles"] == self.target_df["smiles"])

        self.qnn_feats = self.qnn_df.iloc[:, 1:].values
        self.ffnn_feats = self.ffnn_df.iloc[:, 1:].values
        self.targets = self.target_df.iloc[:, 1:].values

    def __len__(self):
        return len(self.qnn_feats)

    def __getitem__(self, idx):
        q = torch.tensor(self.qnn_feats[idx])
        c = torch.tensor(self.ffnn_feats[idx])
        t = torch.tensor(self.targets[idx])
        return q, c, t

# ===== Hybrid Model (QNN + FFNN) =====
class QFFCombiner(nn.Module):
    def __init__(self, dim):
        super(QFFCombiner, self).__init__()
        self.s_q = nn.Parameter(torch.full((dim,), 0.5))
        self.s_c = nn.Parameter(torch.full((dim,), 0.5))

    def forward(self, q, c):
        return self.s_q * q + self.s_c * c

# ===== Training Function =====
def train(qnn_path, ffnn_path, target_path, device='cpu', num_epochs=50, batch_size=32):
    dataset = QFFDataset(qnn_path, ffnn_path, target_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    dim = dataset.qnn_feats.shape[1]
    model = QFFCombiner(dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    loss_fn = nn.MSELoss()

    for epoch in range(num_epochs):
        total_loss = 0
        sample_count = 0

        for q, c, t in loader:
            q, c, t = q.to(device), c.to(device), t.to(device)
            pred = model(q, c)
            loss = loss_fn(pred, t)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.item() * t.size(0)
            sample_count += t.size(0)
        avg_loss = total_loss / sample_count
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.9f}")



if __name__ == '__main__':
    qnn_file = './output/model/qh2_2100_layer3/fold_0/qh2_2100_layer3.csv'
    ffnn_file = './output/model/classical_2100_layer3/fold_0/classical_2100_layer3.csv'
    target_file = './data/research_data/test_full.csv'
    train(qnn_file, ffnn_file, target_file, device='cuda')
