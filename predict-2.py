'''
实现了一个双层神经网络。基于一个字符预测下一个字符。
第一层包含一个线性变换以及一个激活函数。
在激活函数之前修改了线性层的权重使得分布方差依然为1。
第二层是输出层，包含一个线性变换。
token编码映射到一个二维空间。


总结：
1. 上下文太短了，只能看到前一个字符。
2. 参数标准化的时候，别忘了把self.b初始化为zero。
3. 学习率0.1对AdamW来说太大了。默认是1e-3。
'''


import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

BATCH_SIZE = 20
EMBED_SIZE = 2


class myLinear(nn.Module):
	def __init__(self, in_features, hidden_features):
		super().__init__()
		# Error-3: remember to call super().__init__() at first, otherwise the parameters will not be correctly set
		self.W = nn.Parameter(
			torch.randn(in_features, hidden_features) / math.sqrt(in_features)
		) 
		# Error-1: remember to set them as parameters!!
		self.b = nn.Parameter(
			torch.zeros(hidden_features)
		)
		# Error-6: self.b is better initialized as zero

	
	def forward(self, x):
		return x @ self.W + self.b
	
class myTanh(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x):
		exp_2x = torch.exp(2 * x)
		negative = (exp_2x - 1) / (exp_2x + 1)
		exp_minus_2x = torch.exp(-2 * x)
		positive = (1 - exp_minus_2x) / (1 + exp_minus_2x)
		return torch.where(x >= 0, positive, negative)
		# Error-7: Exp(x) is inf if x is too large

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
	one input character to one output character
	"""

	X_train = []
	Y_train = []
	for name in dataset:
		X_train.append("$")
		Y_train.append(name[0])
		# print(f"$ -> {name[0]}")
		for x, y in zip(name, name[1:]):
			X_train.append(x)
			Y_train.append(y)
			# print(f"{x} -> {y}")
		X_train.append(name[-1])
		Y_train.append("$")
		# print(f"{name[-1]} -> $")
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
	x_label = torch.tensor([stoi[i] for i in input_])
	y_label = torch.tensor([stoi[o] for o in output_])
	return x_label, y_label

def generate(embedding, model, encoder, decoder):
	x = '$'
	res = ""

	while True:
		x_label = torch.tensor([encoder[x]])
		x_embedding = embedding(x_label) # 1 * C
		logits = model(x_embedding) # 1 * V
		probs = F.softmax(logits, dim=1)
		# print(probs)

		# sample the next token based on the probs
		# print(probs[0])
		m = torch.distributions.Categorical(probs[0])
		sample = m.sample().item()
		# print(f"sample {sample}")

		# decode the next character
		x = decoder[sample]

		# reach the end
		if x == '$':
			break

		# append x
		res += x
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
	model = myNetwork(EMBED_SIZE, vocab_size)
	optimizer = torch.optim.AdamW(
		chain(
			embedding.parameters(),
			model.parameters()
		),
		lr = 0.01
		# Error-7: learning rate 0.1 is too large for Adam
	)
	# Error-2: Do not forget to initialize the optimizer as well!!
	# Error-5!!!!The embedding's parameter is forgot???


	# train
	for i in range(10000):
		# sample
		idx = torch.randint(len(X_label), (BATCH_SIZE, ))
		# Error-4: the size (position 2) must be a tuple!
		X_sample = X_label[idx]
		Y_sample = Y_label[idx]

		# foward once
		X_embedding = embedding(X_sample)
		logits = model(X_embedding)

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
# 10.431485176086426
# 2.745993137359619
# 2.90327787399292
# 2.5808308124542236
# 2.3906614780426025
# 2.373059034347534
# 2.731419563293457
# 2.443342447280884
# 2.416532039642334
# 2.0886666774749756

# prediction:
# thydalyanilelilijw
# ga
# selon
# wehalelemas
# sh
# kxtan
# mh
# k
# eceilon
# daseefii

# after normalization: dividing self.W by math.sqrt(fan_in)
# loss:
# 3.41955828666687
# 2.3978731632232666
# 2.347989320755005
# 2.8034250736236572
# 2.5115668773651123
# 2.7252211570739746
# 2.683234930038452
# 3.0204710960388184
# 2.523759365081787
# 2.3437743186950684

# prediction
# ponnaiosshinsftl
# aylanolyuwa
# sann
# ryn
# vurola
# ahomalegiyslariwrirelash
# kish
# chinaneyvaranar
# imeosorinidrynyeso
# jans