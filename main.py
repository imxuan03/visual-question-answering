import os
import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel, ViTFeatureExtractor, BertTokenizer, BertModel
from PIL import Image
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Load ViT model and feature extractor
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model_vit = ViTModel.from_pretrained('google/vit-base-patch16-224')

# Load BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

# Define dataset class
class VQADataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.le = LabelEncoder()
        self.data['encoded_answer'] = self.le.fit_transform(self.data['answer'])  # Mã hóa câu trả lời

        # Kiểm tra nếu dataset có dữ liệu
        if len(self.data) == 0:
            raise ValueError("Dataset is empty. Check your CSV file.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Load image and extract features using ViT
        image_path = row['image_path']
        image = Image.open(image_path).convert("RGB")
        inputs = feature_extractor(images=image, return_tensors="pt")
        image_features = model_vit(**inputs).last_hidden_state[:, 0, :]  # Lấy đặc trưng đầu tiên

        # Tokenize question and extract features using BERT
        question = row['question']
        inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True)
        question_features = model_bert(**inputs).last_hidden_state[:, 0, :]  # Lấy đặc trưng đầu tiên

        # Combine image and question features
        combined_features = torch.cat((image_features, question_features), dim=1).squeeze()

        # Encode label
        label = torch.tensor(self.data.iloc[idx]['encoded_answer'], dtype=torch.long)
        return combined_features, label

# Define VQA model class
class VQAModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VQAModel, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
    
    def forward(self, combined_features):
        x = self.fc1(combined_features)
        x = self.relu(x)
        x = self.fc2(x)
        return x  # Không cần sử dụng softmax ở đây vì CrossEntropyLoss đã xử lý

# Load dataset
csv_file = '/kaggle/working/vqa_dataset.csv'
dataset = VQADataset(csv_file)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# Initialize model
input_dim = 768 + 768  # 768 cho ViT và 768 cho BERT
hidden_dim = 512
output_dim = len(dataset.data['encoded_answer'].unique())  # Số lớp = số câu trả lời khác nhau
model_vqa = VQAModel(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_vqa.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_vqa.to(device)

# Training loop
num_epochs = 3
train_losses = []

for epoch in range(num_epochs):
    model_vqa.train()
    epoch_loss = 0.0
    for i, (features, labels) in enumerate(dataloader):
        features = features.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        output = model_vqa(features)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

        if (i + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Step [{i + 1}/{len(dataloader)}], Loss: {loss.item():.4f}')
    
    train_losses.append(epoch_loss / len(dataloader))

print("Training complete.")

# Plot loss
plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss vs Epochs')
plt.legend()
plt.show()

# Evaluate model (example evaluation function)
def evaluate_model(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f"Accuracy: {accuracy:.2f}%")

evaluate_model(model_vqa, dataloader)
