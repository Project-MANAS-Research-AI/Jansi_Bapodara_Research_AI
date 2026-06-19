import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First Conv layer of the block
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second Conv layer of the block
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection (Identity mapping)
        self.shortcut = nn.Sequential()
        
        # If dimensions mismatch (stride != 1 or channels change), adjust the shortcut shape
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        # Main path
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Add the shortcut connection before applying the final ReLU
        out += self.shortcut(x)
        out = F.relu(out)
        return out

  class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_channels = 64

        # Modified Stem for CIFAR-10: 3x3 conv, stride 1, padding 1. No MaxPool!
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # The 4 core layers of ResNet-18 (each containing 2 residual blocks)
        self.layer1 = self._make_layer(BasicBlock, 64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, num_blocks=2, stride=2)
        
        # Classification Head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, num_blocks, stride):
        # The first block handles downsampling if stride == 2
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_channels, out_channels, s))
            self.in_channels = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.linear(out)
        return out


import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

def get_data_loaders(batch_size=128):
    # Training augmentations
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Testing transforms (No flipping or cropping)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader


import torch.optim as optim
import matplotlib.pyplot as plt

def train_and_evaluate():
    # Hardware check
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # Initialize data, network, loss, optimizer, and scheduler
    trainloader, testloader = get_data_loaders(batch_size=128)
    model = ResNet18(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # Paper-aligned hyperparameters with adjusted CIFAR weight decay
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150)

    # Metric tracking lists
    train_losses = []
    test_accuracies = []
    epochs = 150

    for epoch in range(epochs):
        # --- TRAINING PHASE ---
        model.train()
        running_loss = 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
        epoch_loss = running_loss / len(trainloader.dataset)
        train_losses.append(epoch_loss)

        # --- EVALUATION PHASE ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        epoch_acc = (correct / total) * 100
        test_accuracies.append(epoch_acc)

        # Update learning rate
        scheduler.step()

        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {epoch_loss:.4f} - Test Acc: {epoch_acc:.2f}%")

    # --- PLOTTING & SAVING DELIVERABLES ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot Training Loss
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color='tab:red')
    ax1.plot(range(1, epochs + 1), train_losses, color='tab:red', label='Train Loss')
    ax1.tick_params(axis='y', labelcolor='tab:red')

    # Instantiate a second axes that shares the same x-axis for accuracy
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Test Accuracy (%)', color='tab:blue')
    ax2.plot(range(1, epochs + 1), test_accuracies, color='tab:blue', label='Test Acc')
    ax2.tick_params(axis='y', labelcolor='tab:blue')
    
    # 90% Goal Line
    ax2.axhline(y=90, color='green', linestyle='--', alpha=0.7, label='90% Target')

    fig.tight_layout()
    plt.title('ResNet-18 from Scratch on CIFAR-10')
    plt.savefig('resnet18_cifar10_performance.png')
    plt.show()
    print("Training finished! Plot saved as 'resnet18_cifar10_performance.png'")

if __name__ == '__main__':
    train_and_evaluate()
