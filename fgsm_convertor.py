import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os

# ========== PARAMETERS ==========
original_folder = "original_dataset"
adversarial_folder = "adversarial_dataset"
BATCH_SIZE = 64
EPSILON = 0.05   # perturbation strength
LEARNING_RATE = 0.001
NUM_EPOCHS = 20
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ========== LOAD & PREPROCESS DATA ==========
for csv_file in os.listdir(original_folder):
    if csv_file.endswith('.csv'):
        csv_path = os.path.join(original_folder, csv_file)
        adv_csv_file = csv_file.replace('_labeled', '_fgsm')
        adv_csv_path = os.path.join(adversarial_folder, adv_csv_file)
        print(f"Processing {csv_file}...")
        print("Loading dataset...")
        df = pd.read_csv(csv_path)

        # Drop non-numeric / ID / timestamp columns
        drop_cols = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        # Handle null / inf values
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.dropna(inplace=True)

        # Separate features and labels
        labels = df["Label"]
        features = df.drop(columns=["Label"])

        # Encode labels (BENIGN=0, others=1 for binary classification)
        encoder = LabelEncoder()
        labels_enc = encoder.fit_transform(labels)

        # Normalize numeric features
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features.values)

        # Final dataset as numpy arrays
        X = features_scaled.astype(np.float32)
        y = labels_enc.astype(np.int64)


        # ========== DATASET & DATALOADER ==========
        class CSVDataset(Dataset):
            def __init__(self, X, y):
                self.X = torch.tensor(X, dtype=torch.float32)
                self.y = torch.tensor(y, dtype=torch.long)

            def __len__(self):
                return len(self.y)

            def __getitem__(self, idx):
                return self.X[idx], self.y[idx]


        dataset = CSVDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


        # ========== SIMPLE MODEL ==========
        class SimpleNet(nn.Module):
            def __init__(self, input_dim, num_classes):
                super(SimpleNet, self).__init__()
                self.fc1 = nn.Linear(input_dim, 128)
                self.relu = nn.ReLU()
                self.fc2 = nn.Linear(128, num_classes)

            def forward(self, x):
                out = self.fc1(x)
                out = self.relu(out)
                out = self.fc2(out)
                return out


        input_dim = X.shape[1]
        num_classes = len(np.unique(y))
        model = SimpleNet(input_dim, num_classes).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)


        # ========== TRAIN MODEL ==========
        print("Training model...")
        model.train()
        for epoch in range(NUM_EPOCHS):
            epoch_loss = 0.0
            for data, target in dataloader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {epoch_loss/len(dataloader):.4f}")


        # ========== FGSM ATTACK ==========
        def fgsm_attack(data, epsilon, data_grad):
            sign_data_grad = data_grad.sign()
            perturbed = data + epsilon * sign_data_grad
            perturbed = torch.clamp(perturbed, 0.0, 1.0)  # keep in [0,1] range
            return perturbed


        print("Generating adversarial dataset...")
        model.eval()

        adv_examples = []
        adv_labels = []

        for i in range(len(dataset)):
            data, target = dataset[i]
            data, target = data.to(DEVICE), target.to(DEVICE)

            data = data.unsqueeze(0)  # add batch dimension
            data.requires_grad = True

            output = model(data)
            loss = criterion(output, target.unsqueeze(0))
            model.zero_grad()
            loss.backward()

            data_grad = data.grad.data
            perturbed_data = fgsm_attack(data, EPSILON, data_grad)

            adv_examples.append(perturbed_data.detach().cpu().numpy().squeeze())
            adv_labels.append(target.item())

        adv_features = np.array(adv_examples)
        adv_labels = np.array(adv_labels).reshape(-1, 1)

        # Inverse transform to original scale
        adv_features_orig = scaler.inverse_transform(adv_features)

        # Rebuild DataFrame with same columns
        adv_df = pd.DataFrame(adv_features_orig, columns=features.columns)
        adv_df["Label"] = encoder.inverse_transform(adv_labels.flatten())

        # Save adversarial dataset
        adv_df.to_csv(adv_csv_path, index=False)
        print(f"Adversarial dataset saved to {adv_csv_path}")
