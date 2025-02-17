import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
#定义LeNet-5网络
class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.layers = nn.ModuleList([
            nn.Conv2d(1,6,kernel_size=5), # input (batch,1,28,28), output (batch,6,24,24)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2), # output (batch,6,12,12)
            nn.Conv2d(6,16,kernel_size=5) , # output (batch,16,8,8)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2,stride=2), # output (batch,16,4,4)
            nn.Flatten(),  # output (batch,16*4*4)
            nn.Linear(16*4*4, 120), #output (batch,120)
            nn.ReLU(),
            nn.Linear(120,84),
            nn.ReLU(),
            nn.Linear(84,10)
        ])

    def forward(self,x):
        #print(f"Input shape:{x.shape}")
        for layer in self.layers:
            x = layer(x)
            #print(f"After {layer.__class__.__name__}: {x.shape}")
        return x

# load MNIST dataset
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data',train=True,download=True,transform=transform)
test_dataset = datasets.MNIST(root='./data',train=False,download=True,transform=transform)
train_loader = DataLoader(train_dataset,batch_size=32,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=1000,shuffle=False)

# instantiate the model
model = LeNet5()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# train the model and plot loss curve
num_epochs = 10
train_losses = []
test_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    start_time = time.time()  # Record start time
    model.train()
    running_loss = 0.0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()  # zero the gradients
        outputs = model(inputs)

        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)  # Accumulate loss

    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    # evaluate the model on the test set
    model.eval()
    test_loss = 0.0
    correct = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            test_loss += loss.item() * inputs.size(0)
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    accuracy = 100. * correct / len(test_loader.dataset)
    test_accuracies.append(accuracy)

    epoch_time = time.time() - start_time  # Calculate elapsed time
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {epoch_loss:.4f}, Test Loss: {test_loss:.4f}, Test Accuracy: {accuracy:.2f}%, Time: {epoch_time:.2f} seconds")

# Plot the loss curves
epochs = range(1, num_epochs + 1)
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, test_losses, label='Test Loss')
# Annotate the loss values on the plot
for i, (train_loss, test_loss) in enumerate(zip(train_losses, test_losses)):
    plt.text(i + 1, train_loss, f'{train_loss:.4f}', ha='center', va='bottom')
    plt.text(i + 1, test_loss, f'{test_loss:.4f}', ha='center', va='bottom')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss Curves')
plt.legend()
plt.show()