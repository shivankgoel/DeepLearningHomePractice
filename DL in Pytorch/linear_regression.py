import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

x_values = [i for i in range(11)]
x_train = np.array(x_values, dtype=np.float32)
x_train = x_train.reshape(-1, 1)

y_values = [2*i + 1 for i in x_values]
y_train = np.array(y_values, dtype=np.float32)
y_train = y_train.reshape(-1, 1)

'''
CREATE MODEL CLASS
'''
class MyLinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MyLinearRegressionModel, self).__init__()
        self.mylinearfunction = nn.Linear(input_dim, output_dim)  
    
    def forward(self, x):
        out = self.mylinearfunction(x)
        return out

'''
INSTANTIATE MODEL CLASS
'''
input_dim = 1
output_dim = 1

model = MyLinearRegressionModel(input_dim, output_dim)
criterion = nn.MSELoss()
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 100
inputs = Variable(torch.from_numpy(x_train))
labels = Variable(torch.from_numpy(y_train))
for epoch in range(epochs):
    optimizer.zero_grad() 
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print('epoch {}, loss {}'.format(epoch, loss.data[0]))




'''
Note: Optimizer updates the parameters 
using gradient stored in them

list(model.parameters())[0].grad
Variable containing:
-144.4616
[torch.FloatTensor of size 1x1]

>>> list(model.parameters())[1].grad
Variable containing:
-21.5512
[torch.FloatTensor of size 1]
'''