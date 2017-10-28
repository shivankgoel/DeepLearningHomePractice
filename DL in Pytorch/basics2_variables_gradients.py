import torch
from torch.autograd import Variable

a = Variable(torch.ones(2,2), required_grad = True)
x = Variable(torch.ones(2), required_grad = True)
y = 5*((x+1)**2)
o = 0.5* torch.sum(y)
#It calculates gradients of output with respect to all the inputd on which o depends
o.backward()
# To sccess gradients with respect to x
x.grad

