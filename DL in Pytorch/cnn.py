import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable

'''
STEP 1: LOADING DATASET
'''
train_dataset = dsets.MNIST(root='./data', 
                            train=True, 
                            transform=transforms.ToTensor(),
                            download=True)

test_dataset = dsets.MNIST(root='./data', 
                           train=False, 
                           transform=transforms.ToTensor())

batch_size = 100
n_iters = 3000
num_epochs = int((n_iters * batch_size ) / len(train_dataset))


train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

'''
STEP 3: CREATE MODEL CLASS
'''
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()     
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2)
        self.relu1 = nn.ReLU()
        #Other pooling options are AvgPool2d
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.cnn3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(32 * 32 * 32, 5) 
    
    def forward(self, x):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        out = self.maxpool1(out)
        out = self.cnn2(out)
        out = self.relu2(out)
        out = self.maxpool2(out)
        out = self.cnn3(out)
        out = self.relu3(out)
        out = self.maxpool3(out)
        # Resize , for batch size 100
        # Original size: (100, 32, 7, 7)
        # out.size(0): 100
        # New out size: (100, 32*7*7)
        out = out.view(out.size(0), -1)
        # Linear function (readout)
        out = self.fc1(out)
        
        return out

model = CNNModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        
        #Note no need to resize as images.view(-1,28*28) 
        images = Variable(images)
        labels = Variable(labels)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        iter += 1        
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images)
                outputs = model(images)
                predicted_values, predicted_index = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted_index == labels).sum()
            accuracy = 100.0 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))


'''
Iteration: 500. Loss: 0.149548128247. Accuracy: 97.42
Iteration: 1000. Loss: 0.0331667847931. Accuracy: 98.23
Iteration: 1500. Loss: 0.0332339070737. Accuracy: 98.49
Iteration: 2000. Loss: 0.0337009765208. Accuracy: 98.73
Iteration: 2500. Loss: 0.0168377012014. Accuracy: 98.69
Iteration: 3000. Loss: 0.0388985238969. Accuracy: 98.78
'''