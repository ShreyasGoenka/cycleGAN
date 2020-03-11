import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleGenerator(nn.Module):

	def __init__(self):
		super(SimpleGenerator, self).__init__();
		self.fc1 = nn.Linear(2048, 1024);
		self.fc2 = nn.Linear(1024, 1536);
		self.fc3 = nn.Linear(1536, 2048);

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class SimpleDiscriminator(nn.Module):

	def __init__(self):
		super(SimpleDiscriminator, self).__init__()
		self.fc1 = nn.Linear(2048, 1024)
		self.fc2 = nn.Linear(1024, 1024)
		self.fc3 = nn.Linear(1024, 1)

	def forward(self, x):
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

class DropoutG(nn.Module):

	def __init__(self):
		super(DropoutG, self).__init__();
		self.fc1 = nn.Linear(2048, 1024);
		self.dp1 = nn.Dropout()
		self.fc2 = nn.Linear(1024, 1536);
		self.dp2 = nn.Dropout()
		self.fc3 = nn.Linear(1536, 2048);

	def forward(self, x):
		x = self.dp1(F.relu(self.fc1(x)))
		x = self.dp2(F.relu(self.fc2(x)))
		x = self.fc3(x)
		return x