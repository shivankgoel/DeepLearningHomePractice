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

class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return out

input_dim = 28*28
output_dim = 10
model = LogisticRegressionModel(input_dim, output_dim)
criterion = nn.CrossEntropyLoss()


learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

iter = 0
for epoch in range(num_epochs):
    #Enumerate doesn't give us first element of train_loader
    #Enumerate returns tuple (i,element) where i is enumeration number
    for i, (images, labels) in enumerate(train_loader):
        #images is a 100 * 728 matrix after this command    
        images = Variable(images.view(-1, 28*28))
        labels = Variable(labels)
        optimizer.zero_grad()
        #note each training input is in a single row 
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        iter += 1     
        if iter % 500 == 0:
            correct = 0
            total = 0
            for images, labels in test_loader:
                images = Variable(images.view(-1, 28*28))
                outputs = model(images)
                #Note .data is used to convert from variable to tensor
                #Since we will be comparing with labels which is tensor
                predicted_values, predicted_index = torch.max(outputs.data, 1)
                
                # Total number of labels
                total += labels.size(0)
                correct += (predicted_index == labels).sum()

            accuracy = 100 * correct / total
            print('Iteration: {}. Loss: {}. Accuracy: {}'.format(iter, loss.data[0], accuracy))