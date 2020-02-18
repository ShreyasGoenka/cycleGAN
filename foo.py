import torch
import torch.nn as nn

class Linear(nn.Module):

	def __init__(self):
		super(Linear, self).__init__()
		self.fc = nn.Linear(3,1);

	def forward(self, x):
		return self.fc(x)

net = Linear()
print(net.fc.weight)
print(net.fc.bias)

optimizer = torch.optim.Adam(net.parameters(), lr = 1)

x = torch.ones([3])
x[0] = 0

loss = net(x) # Batch training ??
print(loss)
