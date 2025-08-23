import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import os

# Custom Dataset for mobility data
class MobilityDataset(Dataset):
    def __init__(self, X, y, uids):
        self.X = torch.FloatTensor(X)  # Shape: (num_samples, seq_length, input_size)
        self.y = torch.FloatTensor(y)  # Shape: (num_samples, input_size)
        self.uids = torch.LongTensor(uids)  # Shape: (num_samples,)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.uids[idx]

# Preprocessing function
def preprocess_data(df, seq_length=48, grid_size=200, output_dir="preprocessed"):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Filter out invalid coordinates (e.g., (0, 0) if it indicates missing data)
    df = df[(df['x'] != 0) | (df['y'] != 0)].copy()
    
    # Normalize coordinates
    df['x_norm'] = df['x'] / grid_size
    df['y_norm'] = df['y'] / grid_size
    
    # Create time index
    df['time_idx'] = df['d'] * 48 + df['t']
    
    # Encode user IDs
    le = LabelEncoder()
    df['uid_encoded'] = le.fit_transform(df['uid'])
    
    # Initialize lists for sequences
    X, y, uids = [], [], []
    
    # Group by user
    for uid in df['uid_encoded'].unique():
        df_user = df[df['uid_encoded'] == uid].sort_values('time_idx')
        coords = df_user[['x_norm', 'y_norm']].values
        time_idx = df_user['time_idx'].values
        
        # Create sequences
        for i in range(len(coords) - seq_length):
            if time_idx[i + seq_length] == time_idx[i] + seq_length:  # Ensure continuous
                X.append(coords[i:i + seq_length])
                y.append(coords[i + seq_length])
                uids.append(uid)
    
    X = np.array(X)
    y = np.array(y)
    uids = np.array(uids)
    
    # Save preprocessed data
    np.save(os.path.join(output_dir, 'X.npy'), X)
    np.save(os.path.join(output_dir, 'y.npy'), y)
    np.save(os.path.join(output_dir, 'uids.npy'), uids)
    np.save(os.path.join(output_dir, 'label_encoder.npy'), le.classes_)
    
    return X, y, uids, le

# GRU Model with user embedding
class GRUMobilityPredictor(nn.Module):
    def __init__(self, input_size=2, hidden_size=32, num_layers=1, num_users=10000):
        super(GRUMobilityPredictor, self).__init__()
        self.user_embedding = nn.Embedding(num_users, 8)  # Embed user IDs
        self.gru = nn.GRU(input_size + 8, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, input_size)
    
    def forward(self, x, uids):
        # Embed user IDs
        user_emb = self.user_embedding(uids)  # Shape: (batch_size, embed_dim)
        user_emb = user_emb.unsqueeze(1).expand(-1, x.size(1), -1)  # Match sequence length
        x = torch.cat((x, user_emb), dim=-1)  # Concatenate with input
        _, hidden = self.gru(x)
        out = self.fc(hidden[-1])
        return out

# Training function
def train_model(model, dataloader, device, epochs=50, lr=0.001):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y, batch_uids in dataloader:
            batch_X, batch_y, batch_uids = batch_X.to(device), batch_y.to(device), batch_uids.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X, batch_uids)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch + 1}, Avg Loss: {total_loss / len(dataloader):.4f}')
    
    # Save model
    torch.save(model.state_dict(), 'gru_model.pth')

# Prediction function
def predict(model, dataloader, device, steps=720, grid_size=200):
    model = model.to(device)
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for batch_X, _, batch_uids in dataloader:
            batch_X, batch_uids = batch_X.to(device), batch_uids.to(device)
            current_seq = batch_X.clone()
            batch_preds = []
            
            for _ in range(steps):
                pred = model(current_seq, batch_uids)
                batch_preds.append(pred.cpu().numpy())
                current_seq = torch.cat((current_seq[:, 1:, :2], pred.unsqueeze(1)), dim=1)
            
            batch_preds = np.array(batch_preds).transpose(1, 0, 2) * grid_size  # Denormalize
            for i, uid in enumerate(batch_uids.cpu().numpy()):
                predictions[uid] = batch_preds[i]
    
    return predictions

# Main execution
def main():
    # Load data (replace with your CSV file)

    
    # Preprocess data
    seq_length = 48  # One day
    X, y, uids, label_encoder = preprocess_data(data, seq_length=seq_length)
    
    # Create dataset and dataloader
    dataset = MobilityDataset(X, y, uids)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=4)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GRUMobilityPredictor(input_size=2, hidden_size=32, num_layers=1, num_users=len(label_encoder.classes_))
    
    # Train model
    train_model(model, dataloader, device, epochs=50)
    
    # Predict for days 61–75 (720 time steps)
    pred_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)
    predictions = predict(model, pred_dataloader, device, steps=720)
    
    # Decode predictions
    uid_mapping = {i: uid for i, uid in enumerate(label_encoder.classes_)}
    for uid_encoded, preds in predictions.items():
        uid = uid_mapping[uid_encoded]
        print(f"User {uid} predictions (x, y):", preds[:5])  # Show first 5 predictions

if __name__ == "__main__":
    main()