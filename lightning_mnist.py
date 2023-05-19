import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import MNIST
from torchvision import transforms
!pip install pytorch-lightning
import pytorch_lightning as pl

class LitNN(pl.LightningModule):
	def __init__(self):
		super().__init__()
		self.lyrs = nn.Sequential(nn.Linear(28 * 28, 64),nn.ReLU(),nn.Linear(64, 10))

	def forward(self, x):
		out = self.lyrs(x)
		return out

	def configure_optimizers(self):
		optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
		return optimizer

	def training_step(self, train_batch, batch_idx):
		images, labels = train_batch
		images = images.view(images.size(0), -1)
		outputs = self.lyrs(images)
		loss = F.cross_entropy(outputs, labels)
		self.log('train_loss', loss)
		return loss

	def validation_step(self, val_batch, batch_idx):
		images, labels = val_batch
		images = images.view(images.size(0), -1)
		outputs = self.lyrs(images)
		loss = F.cross_entropy(outputs, labels)
		self.log('val_loss', loss)

# data
dataset = MNIST('', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=32)

# model
model = LitNN()

# training
trainer = pl.Trainer(max_epochs=1)
trainer.fit(model, train_loader, val_loader)

print(model)