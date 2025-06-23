import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_parser import DataParser, BrainTumorDataset, split_dataset
import torch.optim as optim
from tqdm import tqdm
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import gc

img_size = (224, 224)
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

print(train_df['label'].value_counts())

label_map = {label: idx for idx, label in enumerate(train_df["label"].unique())}

train_dataset = BrainTumorDataset(train_df, label_map, transform)
val_dataset = BrainTumorDataset(val_df, label_map, transform)
#test_dataset = BrainTumorDataset(test_df, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)


class AEClassifier(nn.Module):
    def __init__(self, img_size, sizes, num_classes=2):
        super(AEClassifier, self).__init__()
        self.img_size = img_size
        self.encoder = nn.Sequential(
            # 1st layer
            nn.Conv2d(1, sizes[0], kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(sizes[0]),
            nn.ReLU(),
            
            # 2nd layer
            nn.Conv2d(sizes[0], sizes[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(sizes[1]),
            nn.ReLU(),
            
            # 3rd layer
            nn.Conv2d(sizes[1], sizes[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(sizes[2]),
            nn.ReLU(),
            
            # 4th layer
            nn.Conv2d(sizes[2], sizes[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(sizes[3]),
            nn.ReLU()
            # now we have latent representation
        )
        # same but in reverse
        self.decoder = nn.Sequential(
            # 1st layer
            nn.ConvTranspose2d(sizes[3], sizes[2], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(sizes[2]),
            nn.ReLU(),
            # 2st layer
            nn.ConvTranspose2d(sizes[2], sizes[1], kernel_size=3, stride=2, padding=1, output_padding=1), 
            nn.BatchNorm2d(sizes[1]),
            nn.ReLU(),
            # 3st layer
            nn.ConvTranspose2d(sizes[1], sizes[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(sizes[0]),
            nn.ReLU(),
            # 4st layer
            nn.ConvTranspose2d(sizes[0], 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
        
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(sizes[3], sizes[2]),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(sizes[2], num_classes)
        )

    def forward(self, x):
        x = self.encoder(x)
        l = self.classifier(x)
        r = self.decoder(x) 
        return r, l

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

list_embed = [
    #[256, 128, 64, 32],
    [128, 64, 32, 16],
    [2048, 1024, 512, 256]
]

idx = 1

for embed_sizes in list_embed:
    
    torch.cuda.empty_cache() 
    idx += 1
    num_classes = len(set(sample[1] for sample in train_dataset))

    print("CLASSES:", num_classes)
    model = AEClassifier(img_size=img_size, sizes = embed_sizes, num_classes=num_classes).to(device)

    # loss functions
    recon_crt = nn.MSELoss()
    class_crt = nn.CrossEntropyLoss()

    # optimizer
    #optimizer = optim.Adam(model.parameters(), lr=0.001)
    optimizer = optim.Adam(model.parameters(), lr=0.01)#, weight_decay=1e-5)

    num_epochs = 20

    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    def train(model, train_loader, val_loader, optimizer, num_epochs):
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            total_class = 0 
            total_recon = 0
            correct = 0
            total = 0
            
            loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)
            
            for images, labels in loop:
                images, labels = images.to(device), labels.to(device)
                reconstructed, logits = model(images)

                recon_loss = recon_crt(reconstructed, images)#.view(images.size(0), -1))
                class_loss = class_crt(logits, labels)
                #combined loss reconstruction + class
                loss = recon_loss + class_loss 

                optimizer.zero_grad()   
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                predictions = torch.argmax(logits, dim=1)
                correct += (predictions == labels).sum().item()
                total += labels.size(0)

                total_loss += loss.item()
                total_class += class_loss.item()
                total_recon += recon_loss.item()

                loop.set_postfix(loss=loss.item(), acc=100 * correct / total)
            
            # Print epoch results
            print(f"Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, "
                f"Class={total_class/len(train_loader):.4f}, "
                f"Recon={total_recon/len(train_loader):.4f}, "
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

    torch.save(model.state_dict(), f"ae_classifier{idx}.pth")
    del model
    torch.cuda.empty_cache()
    gc.collect()