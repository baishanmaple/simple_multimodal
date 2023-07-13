import os
from PIL import Image
import nltk
from nltk import word_tokenize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from torchtext.vocab import build_vocab_from_iterator
import time

data_path = "datafolder/data"  # 替换为数据集所在的文件夹路径
label_mapping = {"positive": 0, "negative": 1, "neutral": 2}

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.text_files = sorted([f for f in os.listdir(data_path) if f.endswith(".txt")])
        self.image_files = sorted([f for f in os.listdir(data_path) if f.endswith(".jpg")])

        assert len(self.text_files) == len(self.image_files), "文本和图像文件数量不匹配"

        self.data = []

        tokenizer = word_tokenize
        self.tokenizer = tokenizer

        label_file = os.path.join("实验五数据/train.txt")
        with open(label_file, "r") as file:
            lines = file.readlines()
            for line_idx, line in enumerate(lines):
                if line_idx == 0:  # 跳过标题行
                    continue
                parts = line.strip().split(",")
                guid = parts[0].strip()
                label = parts[1].strip()

                text_file = guid + ".txt"
                image_file = guid + ".jpg"

                text_path = os.path.join(data_path, text_file)
                image_path = os.path.join(data_path, image_file)

                # 处理文本文件
                with open(text_path, 'r', encoding='utf-8', errors='ignore') as text_file:
                    text_data = text_file.read().strip()

                label = label_mapping[label]
                
                # 添加Example到数据列表中
                self.data.append((text_data, label, image_path))

        # 构建词表（vocabulary）
        self.vocab = build_vocab_from_iterator(self._yield_tokens(), specials=["<unk>", "<pad>", "<sos>", "<eos>"])

        # 图像转换器
        self.image_transform = transforms.Compose([
            # transforms.Resize((224, 224)),  # 调整图像大小为 (224, 224)
            transforms.Resize((200, 200)),
            transforms.ToTensor(),  # 将图像转换为张量
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化图像张量
        ])

    def _yield_tokens(self):
        for text, _, _ in self.data:
            yield self.tokenizer(text)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text_data, label_data, image_path = self.data[index]

        # 将文本转换为Tensor
        tokens = [tok for tok in self.tokenizer(text_data)]
        text_tensor = torch.tensor([self.vocab[token] for token in tokens], dtype=torch.long)

        # 处理图像文件
        image = Image.open(image_path)
        image_tensor = self.image_transform(image)

        return text_tensor, image_tensor, label_data

    def collate_fn(self, batch):
        # 将批次中的文本和图像数据分别提取出来
        texts, images, labels = zip(*batch)

        # 填充文本序列
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=self.vocab["<pad>"])

        # 返回填充后的文本、图像和标签数据
        return padded_texts, torch.stack(images), labels

# 创建自定义数据集实例
dataset = CustomDataset(data_path)
text_vocab_size = len(dataset.vocab)
image_feature_size = 200
p_train = 0.85
train_size = int(p_train * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# 使用DataLoader加载数据集，并指定collate_fn为自定义的方法
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=32, shuffle=True, collate_fn=dataset.collate_fn)
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
# device = torch.device('cpu')
# print(len(dataset.data))

# 定义多模态情感分类模型
class SentimentClassifier(nn.Module):
    def __init__(self, text_input_size, image_input_size, hidden_size, num_classes):
        super(SentimentClassifier, self).__init__()

        # 文本模态的网络层
        self.text_embedding = nn.Embedding(text_input_size, 128)
        self.text_lstm1 = nn.LSTM(128, hidden_size, batch_first=True)
        self.text_lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        
        self.fusion_fc = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=0.5)

        # 最后一层
        self.output_fc = nn.Linear(hidden_size, num_classes)

    def forward(self, text_input, image_input):
        # 文本模态的前向传播
        # [batch size, sequence length]
        text_output = self.text_embedding(text_input)
        # [batch size, sequence length, text embedding]
        # print(text_output.shape)
        text_output, _ = self.text_lstm1(text_output)
        text_output = self.dropout(text_output)
        # print(text_output.shape)
        text_output, _ = self.text_lstm2(text_output)
        # print(text_output.shape)


        output = text_output[:, -1, :]
        output = self.dropout(output)
        output = self.fusion_fc(output)
        output = self.dropout(output)
        output = torch.relu(output)
        
        # 最后一层
        output = self.output_fc(output)
        # output = F.softmax(output, dim=1)
        # print(output)
        # print(output.shape)
        return output


def train(model, train_dataloader, criterion, optimizer, device):
    model.train()  # 设置模型为训练模式
    total_loss = 0.0
    correct = 0
    total = 0

    i = 1
    for texts, images, labels in train_dataloader:
        texts = texts.to(device)
        images = images.to(device)
        labels = torch.tensor(labels).to(device)

        optimizer.zero_grad()

        # 前向传播
        outputs = model(texts, images)
        # mean_values = torch.mean(outputs, dim=1)
        _, predicted = torch.max(outputs, dim=1)
        # print(predicted)
        # print(labels)
        
        # outputs = outputs.view(-1, outputs.shape[-1])
        outputs = torch.squeeze(outputs)
        # print(outputs.shape)
        labels = labels.view(-1)
        # print(labels.shape)
        # print(outputs)
        # print(labels)
        # print(predicted)

        # 计算损失
        loss = criterion(outputs, labels)

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)

        # 统计预测值和标签相等的数量
        correct += torch.sum(predicted == labels).item()
        # print(correct)

        i += 1
        if i % 10 == 0:
            print(f"[ {i*32:4.0f} / {3200} ] Temp Accuracy: {100 * correct / total:.2f}%")

    accuracy = 100 * correct / total
    average_loss = total_loss / len(train_dataloader)

    print(f"Train Loss: {average_loss:.4f} | Accuracy: {accuracy:.2f}%")


def evaluate(model, val_dataloader, criterion, device):
    model.eval()  # 设置模型为评估模式
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for texts, images, labels in val_dataloader:
            texts = texts.to(device)
            images = images.to(device)
            labels = torch.tensor(labels).to(device)

            # 前向传播
            outputs = model(texts, images)
            # mean_values = torch.mean(outputs, dim=1)
            _, predicted = torch.max(outputs, dim=1)

            outputs = torch.squeeze(outputs)
            # print(outputs.shape)
            labels = labels.view(-1)
            # 计算损失
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            total += labels.size(0)

            # 统计预测值和标签相等的数量
            correct += torch.sum(predicted == labels).item()

    accuracy = 100 * correct / total
    average_loss = total_loss / len(val_dataloader)

    print(f"Validation Loss: {average_loss:.4f} | Accuracy: {accuracy:.2f}%")

    return accuracy



# model = SentimentClassifier(text_vocab_size, 224, 224, 100, 3)  # 根据模型架构进行初始化
model = SentimentClassifier(text_vocab_size, 200, 100, 3)  # 根据模型架构进行初始化

# 将模型移动到指定设备上
model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
weight_decay = 0.1
optimizer = optim.Adam(model.parameters(), lr=0.000001)
# optimizer = optim.Adam(model.parameters(), weight_decay=weight_decay)


# 训练和评估循环
num_epochs = 10  # 定义训练循环的迭代次数

best_score = 0
for epoch in range(num_epochs):
    start_time = time.time()
    print(f"Epoch {epoch+1}/{num_epochs}")
    train(model, train_dataloader, criterion, optimizer, device)
    end_time = time.time()
    score = evaluate(model, val_dataloader, criterion, device)
    print('Time Cost of Training: {:.2f} s'.format(end_time-start_time))
    if score > best_score:
        # 保存模型
        torch.save(model.state_dict(), "model.pt")  # 保存模型
