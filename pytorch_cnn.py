import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

device= ('cuda' if torch.cuda.is_available() else 'cpu')
class CNN(nn.Module):
    def __init__(self,in_channel=1,num_classes=10):
        super(CNN,self).__init__()
        self.conv1=nn.Conv2d(in_channels=in_channel,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same padding
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))#haalfs the size
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))#same padding
        self.fc1=nn.Linear(16*7*7,num_classes)

    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=self.pool(x)
        x=F.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc1(x)

        return x

model=CNN().to(device)
# x=torch.randn(64,1,28,28)
# print(model(x).shape)
# exit()
in_channel=1
num_classes=10
learning_rate=0.001
batch_size=64
num_epochs=1
# Data loader
train_dataset=datasets.MNIST(root='pytorch__cnn\datasets',train=True,transform=transforms.ToTensor(),download=True)
train_loader=DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)

test_dataset=datasets.MNIST(root='pytorch__cnn\datasets',train=False,transform=transforms.ToTensor(),download=True)
test_loader=DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)

# Loss and optimizer
criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)


for epochs in range(num_epochs):
    for batch_idx,(data,targets) in enumerate (train_loader):
        data=data.to(device)
        targets=targets.to(device)

        # Forward
        scores=model(data)
        loss=criterion(scores,targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        # Gradient descent or adam step
        optimizer.step()

def check_accuracy(loader,model):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x,y in loader :
            x=x.to(device)
            y=y.to(device)
            # x=x.reshape(x.shape[0],-1)
            scores=model(x)
            _,predictions=scores.max(1)
            num_correct+=(predictions==y).sum()
            num_samples+=predictions.size(0)
        print(f"Got {num_correct}/{num_samples}  with accuracy {num_correct/num_samples*100:.2f}")


check_accuracy(train_loader,model)
check_accuracy(test_loader,model)
