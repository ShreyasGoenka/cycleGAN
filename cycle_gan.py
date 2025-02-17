import torch
import itertools

import networks
import loss
from image_pool import ImagePool
from collections import OrderedDict

# for now assuming that itertools just commbines the parameters of the two networks for simulaneous update, may change later.

class CycleGAN():
	"""This constructs a cycle gan

	parameters:
		loss_names: the loss names to consider while training (maybe even test).
		isTrain: (bool) True if in training mode
		model_names: describes which models to be loaded (no disciminators during testing)
		criterionGAN: describes the adversarial loss for the GANs
		criterionCycle: cycle consistancy loss metric

		optimizers: this has the parameters for G and D, still need to confirm how they alternate between forward and backward.

		
	"""

	def __init__(self, isTrain = True, lr = 0.5, beta1 = 0.0002):

		self.lambda_A = 10.0
		self.lambda_B = 10.0
		self.lr = lr
		self.beta1 = beta1
		self.pool_size=50

		self.loss_names = ['D_A', 'G_A', 'cycle_A', 'D_B', 'G_B', 'cycle_B']
		self.isTrain = isTrain
		self.device =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
		print("device being used: ", self.device)

		self.model_names = []
		self.optimizers = []

		# self.netG_A = networks.SimpleGenerator().to(self.device)
		# self.netG_B = networks.SimpleGenerator().to(self.device)
		self.netG_A = networks.DropoutG().to(self.device)
		self.netG_B = networks.DropoutG().to(self.device)

		if self.isTrain:
			self.model_names = ['G_A', 'G_B', 'D_A', 'D_B']
		else:
			self.model_names = ['G_A', 'G_B']
		
		if self.isTrain:
			self.fake_A_pool = ImagePool(self.pool_size)
			self.fake_B_pool = ImagePool(self.pool_size)

			self.netD_A = networks.SimpleDiscriminator().to(self.device)
			self.netD_B = networks.SimpleDiscriminator().to(self.device)

			# self.criterionGAN = loss.GANLoss('lsgan').to(self.device)  # define GAN loss.
			self.criterionGAN = loss.GANLoss('vanilla').to(self.device)

			self.criterionCycle = torch.nn.L1Loss()

			self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
			# self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr=self.lr, betas=(self.beta1, 0.999))
			# self.optimizer_G = torch.optim.SGD(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr = 0.0001)
			self.optimizer_D = torch.optim.SGD(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters()), lr = 0.0001)
			self.optimizers.append(self.optimizer_G)
			self.optimizers.append(self.optimizer_D)

		self.count = 0
		self.ratio = 1


	def forward(self):
		"""Run forward pass; called by both functions <optimize_parameters> and <test>."""
		# print(self.real_A)
		# print(self.real_A.shape)
		# print(type(self.real_A))
		self.fake_B = self.netG_A(self.real_A)  # G_A(A)
		self.rec_A = self.netG_B(self.fake_B)   # G_B(G_A(A))
		self.fake_A = self.netG_B(self.real_B)  # G_B(B)
		self.rec_B = self.netG_A(self.fake_A)   # G_A(G_B(B))


	def set_input(self, input, direction = 'AtoB'):
		"""Unpack input data from the dataloader and perform necessary pre-processing steps.

		Parameters:
			input (dict): include the data itself and its metadata information.

		The option 'direction' can be used to swap domain A and domain B.
		"""
		AtoB = direction == 'AtoB'
		self.real_A = input['A' if AtoB else 'B'].to(self.device)
		self.real_B = input['B' if AtoB else 'A'].to(self.device)

	def backward_G(self):

		# GAN loss D_A(G_A(A))
		self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
		# GAN loss D_B(G_B(B))
		self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
		# Forward cycle loss || G_B(G_A(A)) - A||
		self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * self.lambda_A
		# Backward cycle loss || G_A(G_B(B)) - B||
		self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * self.lambda_B
		# combined loss and calculate gradients
		self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B
		self.loss_G.backward()


	def backward_D_basic(self, netD, real, fake):
		"""Calculate GAN loss for the discriminator

		Parameters:
			netD (network)      -- the discriminator D
			real (tensor array) -- real images
			fake (tensor array) -- images generated by a generator

		Return the discriminator loss.
		We also call loss_D.backward() to calculate the gradients.
		"""
		# Real
		pred_real = netD(real)
		loss_D_real = self.criterionGAN(pred_real, True)
		# Fake
		pred_fake = netD(fake.detach())
		pred_fake = netD(fake)
		loss_D_fake = self.criterionGAN(pred_fake, False)
		# Combined loss and calculate gradients
		loss_D = (loss_D_real + loss_D_fake) * 0.5
		loss_D.backward()
		return loss_D

	def backward_D_A(self):
		"""Calculate GAN loss for discriminator D_A"""
		fake_B = self.fake_B_pool.query(self.fake_B)
		self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
		# self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)

	def backward_D_B(self):
		"""Calculate GAN loss for discriminator D_B"""
		fake_A = self.fake_A_pool.query(self.fake_A)
		self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, fake_A)
		# self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)



	def optimize_parameters(self):
		"""Calculate losses, gradients, and update network weights; called in every training iteration"""
		# forward
		self.forward()      # compute fake images and reconstruction images.
		# G_A and G_B
		self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
		self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
		# print("before back wt.grad: ", self.netG_A.fc1.weight.grad)
		self.backward_G()             # calculate gradients for G_A and G_B
		# print("before step wt ", self.netG_A.fc1.weight)
		# print(self.netG_A.fc1.weight.shape)
		# print(self.netG_A.fc1.weight.grad.shape)
		self.optimizer_G.step()       # update G_A and G_B's weights
		# print("after back wt.grad: ", self.netG_A.fc1.weight.grad)
		# print("after step wt ", self.netG_A.fc1.weight)	
		# D_A and D_B
		if self.count % self.ratio == 0:
			self.set_requires_grad([self.netD_A, self.netD_B], True)
			self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
			self.backward_D_A()      # calculate gradients for D_A
			self.backward_D_B()      # calculate graidents for D_B
			self.optimizer_D.step()  # update D_A and D_B's weights

		self.count += 1

	def get_current_losses(self):
		"""Return traning losses / errors. train.py will print out these errors on console, and save them to a file"""
		errors_ret = OrderedDict()
		for name in self.loss_names:
			if isinstance(name, str):
				errors_ret[name] = float(getattr(self, 'loss_' + name))  # float(...) works for both scalar tensor and float number
		return errors_ret

	def set_requires_grad(self, nets, requires_grad=False):
		"""Set requies_grad=Fasle for all the networks to avoid unnecessary computations
		Parameters:
			nets (network list)   -- a list of networks
			requires_grad (bool)  -- whether the networks require gradients or not
		"""
		if not isinstance(nets, list):
			nets = [nets]
		for net in nets:
			if net is not None:
				for param in net.parameters():
					param.requires_grad = requires_grad





