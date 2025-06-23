import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from data_parser import DataParser, BrainTumorDataset, split_dataset
from tqdm import tqdm
import numpy as np
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class PosEmb(nn.Module):
    def __init__(self, max_seq, emb_size):
        super().__init__()
        
        pos_emb = torch.zeros(max_seq, emb_size)

        self.cls = nn.Parameter(torch.rand(1, 1, emb_size))
        # sine/cosine trick
        for i in range(max_seq):
            for j in range(emb_size):
                if j % 2 == 0:
                    pos_emb[i][j] = np.sin(i/(10000) ** (j/emb_size))
                else:
                    pos_emb[i][j] = np.cos(i/(10000 ** ((j-1)/emb_size)))

        # register buffer at the end
        self.register_buffer('pos_emb', pos_emb.unsqueeze(0))

    def forward(self, x):
        tokens = self.cls.expand(x.size()[0], -1, -1)

        x = torch.cat((tokens, x), dim = 1) # add cls tokens to each image in the batch
        
        return x + self.pos_emb

class Patches(nn.Module):
    def __init__(self, img_size, emb_size,kernel, channels):
        super().__init__()

        self.conv_layer = nn.Conv2d(channels, emb_size, kernel_size = kernel, stride = kernel)

    def forward(self, x):
        x = self.conv_layer(x) # (Batch X channels x h x w) to (Batch x emd_size x result_h x result_w)
        x = x.flatten(2) # (b x c x rh x rw) to (b x emb_size x result)
        # switch emb_size with result dim
        return x.transpose(1,2)


class Attn(nn.Module):
    def __init__(self, hd_size, emb_size):
        super().__init__()
        
        self.k = nn.Linear(emb_size, hd_size)
        self.v = nn.Linear(emb_size, hd_size)
        self.q = nn.Linear(emb_size, hd_size)
        self.hd = hd_size
    
    def forward(self, x):
        key = self.k(x)
        value = self.v(x)
        query = self.q(x)
        attn = query@key.transpose(-2, -1)
        scaling = (self.hd ** 0.5)
        attn = attn/scaling
        attn = torch.softmax(attn, dim = -1)
        attn = attn @ value
        return attn

class MHA(nn.Module):
    def __init__(self, heads, emb_size):
        super().__init__()
        self.linear = nn.Linear(emb_size, emb_size)
        self.attns = nn.ModuleList([Attn(emb_size // heads, emb_size) for i in range(heads)])
    
    def forward(self, x):
        x = torch.cat([hd(x) for hd in self.attns], dim = -1)
        return self.linear(x)

class Transformer(nn.Module):
    def __init__(self, heads, emb_size, ratio = 4):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_size)
        self.mha = MHA(heads, emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.final_fc = nn.Sequential(nn.Linear(emb_size, emb_size * ratio), nn.GELU(), nn.Linear(emb_size * ratio, emb_size))
    
    def forward(self, x):
        res = x + self.mha(self.norm1(x))
        res = res + self.final_fc(self.norm2(res))

        return res
    
class ViT(nn.Module):
    def __init__(self, classes, img_sz, channels, heads, emb_size, kernel_sz, layers):
        super().__init__()
        self.patches = (img_sz * img_sz) // (kernel_sz * kernel_sz)
        self.max_seq = self.patches + 1
        self.pe = PosEmb(self.max_seq, emb_size)
        self.patches = Patches(img_sz, emb_size, kernel_sz, channels)
        self.transformers = nn.Sequential(*[Transformer(heads, emb_size) for i in range(layers)])
        self.classifier = nn.Linear(emb_size, classes)

    def forward(self, x):
        x = self.patches(x)
        x = self.pe(x)
        x = self.transformers(x)
        return self.classifier(x[:,0])

img_size = (224, 224)
torch.cuda.empty_cache() 
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(img_size)
])

AUG_CONFIG = {
        'flip': {
            'h_prob': 0.7,
            'v_prob': 0.7   
        },
        'rotate': {
            'apply_prob': 0.7, 
            'max_angle': 30
        },
        'brightness': {
            'apply_prob': 0.7, 
            'factor_range': (0.7, 1.3)
        },
        'noise': {
            'apply_prob': 0.7,
            'sigma_limit': 15
        }
}

data_parser = DataParser('./data')
df = data_parser.read_images(img_size=img_size, augmentations=AUG_CONFIG)

train_df, val_df = split_dataset(df)

label_map = {label: idx for idx, label in enumerate(train_df["label"].unique())}

train_dataset = BrainTumorDataset(train_df, label_map, transform)
val_dataset = BrainTumorDataset(val_df, label_map, transform)
#test_dataset = BrainTumorDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
num_classes = len(set(sample[1] for sample in train_dataset))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("CLASSES:", num_classes)
class_crt = nn.CrossEntropyLoss()

img_size = 224
emb_size = 256
classes = num_classes
heads = 4
layers = 4
num_epochs = 40


train_acc = []
val_acc = []
train_loss = []
val_loss = []

model = ViT(classes, img_size, 1, heads, emb_size, 16, layers).to(device)
optimizer = Adam(model.parameters(), lr=0.005)
criterion = nn.CrossEntropyLoss()

def train(model, train_loader, val_loader, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_class_loss = 0
        correct = 0
        total = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = class_crt(logits, labels)

            optimizer.zero_grad()   
            loss.backward()
            optimizer.step()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            total_loss += loss.item()
            total_class_loss += loss.item()

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
        
        print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
              f"Class={total_class_loss/len(train_loader):.4f}, "
              f"Accuracy={100 * correct / total:.2f}%")
        train_acc.append(correct / total)
        train_loss.append(total_loss/len(train_loader))
        validate(model, val_loader)

def validate(model, val_loader):
    model.eval()
    correct = 0
    total = 0
    total_val_loss = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)

            _, logits = model(images)

            class_loss = class_crt(logits, labels)
            total_val_loss += class_loss.item()

            predictions = torch.argmax(logits, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    print(f"Validation: Loss={total_val_loss / len(val_loader):.4f}, "
        f"Accuracy={100 * correct / total:.2f}%")
    val_acc.append(correct/total)
    val_loss.append(total_val_loss / len(val_loader))

train(model, train_loader, val_loader, optimizer, num_epochs)
torch.save(model.state_dict(), "vit_classifier.pth")
print("Model saved successfully!")