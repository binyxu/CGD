import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn

class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
    def __len__(self):
        return len(self.dataset)
    def __getitem__(self, idx):
        data = self.dataset[idx]
        return idx, data  # Return index along with data

class PretrainedClassifier:
    def __init__(self, model, batch_size=32, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.batch_size = batch_size
        self.device = device
        # Load the pretrained model
        self.model = model
        self.model.to(self.device)
        self.model.eval()
        
    def predict(self, dataset):
        dataset = IndexedDataset(dataset)
        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        logits_dict = {}
        with torch.no_grad():
            for idxs, (inputs, labels) in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                outputs = outputs.cpu()
                idxs = idxs.tolist()
                for idx, output in zip(idxs, outputs):
                    logits_dict[idx] = output
        # Now collect logits in order
        logits_list = [logits_dict[idx].unsqueeze(0) for idx in range(len(dataset))]
        logits = torch.cat(logits_list, dim=0)
        return logits
