'''
实现了一个双层神经网络。基于前n个字符预测下一个字符。
第一层包含一个线性变换以及一个激活函数。
在激活函数之前修改了线性层的权重使得分布方差依然为1。
第二层是输出层，包含一个线性变换。
token编码映射到一个二维空间。

'''


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH_SIZE = 20
TIME_SIZE = 3
EMBED_SIZE = 2


class myLinear(nn.Module):
	def __init__(self, in_features, hidden_features):
		super().__init__()
		self.W = nn.Parameter(
			torch.randn(in_features, hidden_features) / math.sqrt(in_features)
		)
		self.b = nn.Parameter(
			torch.randn(hidden_features)
		)

	
	def forward(self, x):
		print("linear:", x.shape)
		return x @ self.W + self.b
	
class myTanh(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		exp_x = torch.exp(x)
		exp_minus_x = torch.exp(-x)
		return (exp_x - exp_minus_x) / (exp_x + exp_minus_x)

class myNetwork(nn.Module):
	def __init__(self, in_features, out_features):
		super().__init__()
		hidden_features = 2 * out_features
		self.network = nn.Sequential(
			myLinear(in_features, hidden_features),
			myTanh(),
			myLinear(hidden_features, out_features),
		)

	def forward(self, x):
		print("network:", x.shape)
		return self.network(x)

class myEmbedding(nn.Module):
	def __init__(self, vocab_size, embed_size):
		super().__init__()
		self.embedding = nn.Parameter(
			torch.randn(vocab_size, embed_size)
		)
	
	def forward(self, x):
		return self.embedding[x]


def acquire_names():
	name_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "names.txt")
	names = open(name_file, "r", encoding="utf-8").read().split("\n")
	return names

def form_training_set(dataset): 
	"""
	three input characters to one output character
	"""

	X_train = []
	Y_train = []
	for name in dataset:
		prefix = '$' * TIME_SIZE
		append_name = prefix + name + '$'
		for i in range(len(append_name) - TIME_SIZE):
			X_train.append(append_name[i: i + TIME_SIZE])
			Y_train.append(append_name[i + TIME_SIZE])
			# print(f"{append_name[i: i + TIME_SIZE]} -> {append_name[i + TIME_SIZE]}")
	return X_train, Y_train

def gen_encoder_decoder(dataset):
	tokens = set()
	for name in dataset:
		for t in name:
			tokens.add(t)
	stoi = {'$': 0}
	itos = {0: '$'}
	tokens = sorted(tokens)
	for t, ind in zip(tokens, range(1, len(tokens) + 1)):
		stoi[t] = ind
		itos[ind] = t
	return stoi, itos

def encode(input_, output_, stoi):
	x_label = torch.tensor([[stoi[ch] for ch in str_] for str_ in input_])
	y_label = torch.tensor([stoi[o] for o in output_])
	return x_label, y_label

def generate(embedding, model, encoder, decoder):
	x = '$' * TIME_SIZE
	res = ""

	while True:
		x_label = torch.tensor([[encoder[ch] for ch in x]]) # 1, T
		x_embedding = embedding(x_label) # 1, T, C
		x_embedding = x_embedding.view(1, -1) # 1, T * C
		logits = model(x_embedding) # 1, O
		probs = F.softmax(logits, dim=1)
		# print(probs)

		# sample the next token based on the probs
		# print(probs[0])
		m = torch.distributions.Categorical(probs[0])
		sample = m.sample().item()
		# print(f"sample {sample}")

		# decode the next character
		y = decoder[sample]

		# reach the end
		if y == '$':
			break

		# append x
		res += y
		x = x[1:] + y
		# Error-1: make sure that x's length is always TIME_SIZE
	return res

def main():

	# load the dataset
	names = acquire_names()
	X_train, Y_train = form_training_set(names)
	encoder, decoder = gen_encoder_decoder(names)
	X_label, Y_label = encode(X_train, Y_train, encoder)

	# initialization
	vocab_size = len(encoder)
	embedding = myEmbedding(vocab_size, EMBED_SIZE)
	model = myNetwork(EMBED_SIZE * TIME_SIZE, vocab_size)
	optimizer = torch.optim.AdamW(
		model.parameters(),
		lr = 0.1
	)
	# Error-2: Do not forget to initialize the optimizer as well!!


	# train
	for i in range(1):
		# sample
		idx = torch.randint(X_label.shape[0], (BATCH_SIZE, ))
		# Error-4: the size (position 2) must be a tuple!
		X_sample = X_label[idx] # B, T
		Y_sample = Y_label[idx] # B

		# foward once
		X_embedding = embedding(X_sample) # B, T, C
		X_embedding = X_embedding.view(BATCH_SIZE, -1) # B, T * C
		logits = model(X_embedding) # B, O

		# calculate the loss and backward
		loss = F.cross_entropy(logits, Y_sample)
		loss.backward()
		if i % 1000 == 0:
			print(loss.item())

		# update the parameters and set the gredients to zero
		optimizer.step()
		optimizer.zero_grad()

	# generate
	for _ in range(10):
		res = generate(embedding, model, encoder, decoder)
		print(res)

main()

# loss:
# 10.303155899047852
# 2.9640934467315674
# 3.1091489791870117
# 3.1857657432556152
# 2.9995250701904297
# 2.7319271564483643
# 2.773827314376831
# 2.247504949569702
# 2.5717110633850098
# 2.9081358909606934

# prediction:
# a
# aninyaeri
# gihg
# kiineeals
# sonamalva
# yoae
# elnionii
# iamexetaemus
# olo

# after normalization: dividing self.W by math.sqrt(fan_in)
# loss:
# 3.700507640838623
# 3.1807751655578613
# 3.142530679702759
# 2.536449909210205
# 3.259244203567505
# 3.281708240509033
# 2.686946392059326
# 3.3252806663513184
# 2.39096736907959
# 2.7931809425354004

# prediction:
# menrs
# js
# panremale
# aeesra
# gahaeeeere
# bahakah
# bva
# je
# arilrereserelanea