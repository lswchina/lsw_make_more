'''
实现了一个三层神经网络。基于前n个字符预测下一个字符。
前两层都分别包含一个线性变换以及一个激活函数。
这两层的结尾位置，手搓了一个batch normalization。
第三层是输出层，包含一个线性变换。
token编码映射到一个五维空间。
功能与上一版本相同，只是这个版本是完全使用pytorch提供的API实现的。


总结：
1. 在网络中，训练和评估时的batch是不一样的。因此，x.view不能统一地把第一个维度设置为BATCH_SIZE。
'''


import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import chain

BATCH_SIZE = 20
TIME_SIZE = 3
EMBED_SIZE = 5
HIDDEN_FEATURES = 50

class myLayer(nn.Module):
	def __init__(self, in_features, hidden_features):
		super().__init__()
		self.layer = nn.Sequential(
			nn.Linear(in_features, hidden_features, bias=False),
			nn.BatchNorm1d(hidden_features),
			nn.Tanh()
		)

	def forward(self, x):
		return self.layer(x)
	
class myNetwork(nn.Module):
	def __init__(self, num_embedding, out_features):
		super().__init__()
		self.embedding = nn.Embedding(num_embedding, EMBED_SIZE)
		self.network = nn.Sequential(
			myLayer(EMBED_SIZE * TIME_SIZE, HIDDEN_FEATURES),
			myLayer(HIDDEN_FEATURES, HIDDEN_FEATURES),
			nn.Linear(HIDDEN_FEATURES, out_features)
		)
	
	def forward(self, x):
		x = self.embedding(x) #B, T, C
		if self.training:
			x = x.view(BATCH_SIZE, -1) #B, T * C
		else:
			x = x.view(1, -1) #1, T * C
		# Error-1: During inference, only one batch is received!
		return self.network(x)


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

def generate(model, encoder, decoder):
	x = '$' * TIME_SIZE
	res = ""

	while True:
		x_label = torch.tensor([[encoder[ch] for ch in x]]) # 1, T
		logits = model(x_label) # 1, O
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
	return res

def main():

	# load the dataset
	names = acquire_names()
	X_train, Y_train = form_training_set(names)
	encoder, decoder = gen_encoder_decoder(names)
	X_label, Y_label = encode(X_train, Y_train, encoder)

	# initialization
	vocab_size = len(encoder)
	model = myNetwork(vocab_size, vocab_size)
	optimizer = torch.optim.AdamW(
		chain(
			model.parameters()
		),
		lr = 0.001
	)


	# train
	model.train()
	for i in range(10000):
		# sample
		idx = torch.randint(X_label.shape[0], (BATCH_SIZE, ))
		X_sample = X_label[idx] # B, T
		Y_sample = Y_label[idx] # B

		# foward once
		logits = model(X_sample) # B, O

		# calculate the loss and backward
		loss = F.cross_entropy(logits, Y_sample)
		loss.backward()
		if i % 1000 == 0:
			print(loss.item())

		# update the parameters and set the gredients to zero
		optimizer.step()
		optimizer.zero_grad()

	# generate
	model.eval()
	for _ in range(10):
		res = generate(model, encoder, decoder)
		print(res)

main()

# loss:
# 3.281569004058838
# 2.6867125034332275
# 2.5172386169433594
# 2.8636066913604736
# 2.611867666244507
# 2.251685619354248
# 2.639862060546875
# 2.7382454872131348
# 2.396827459335327
# 2.0498414039611816

# prediction
# tirkh
# lir
# mami
# alohanit
# malantyhar
# jelana
# adelryn
# jariyahdarranie
# asa
# wer