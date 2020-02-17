import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class Office31(Dataset):

	def __init__(self, csvA, csvB=None):

		self.dataA = np.genfromtxt(csvA, delimiter=',')
		self.featuresA = torch.tensor(self.dataA[:,:2048]).float()
		self.labelsA = self.dataA[:,2048]

		self.dataB = np.genfromtxt(csvB, delimiter=',')
		self.featuresB = torch.tensor(self.dataB[:,:2048]).float()

		print("SELF dataA type", self.featuresA)
		self.labelsB = self.dataB[:,2048]

	def __len__(self):

		return max(self.dataA.shape[0], self.dataB.shape[0])

	def __getitem__(self, index):

		return {'A':self.featuresA[index%self.dataA.shape[0]],
				'B':self.featuresB[index%self.dataB.shape[0]],
				'lA':self.labelsA[index%self.dataA.shape[0]],
				'lB':self.labelsB[index%self.dataB.shape[0]]}


