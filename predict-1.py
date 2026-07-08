'''
实现了一个非常简单的神经网络。
只有一个线性变换层，基于一个字符预测下一个字符。
token编码采用one-hot形式。


结果不算特别好，但至少能看出一些字符两两之间的关系。

总结：
1. 上下文太短了，只能看到前一个字符。
2. 只有线性变换，不能拟合复杂的关系。
3. 神经网络只有一层，能学到的信息有限。
4. 27个token，每一个都用27维向量表示，容易过拟合。
'''


import os
import torch
import torch.nn.functional as F

BATCH_SIZE = 20

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
	x_embedding = F.one_hot(x_label, len(stoi)).float()
	return x_embedding, y_label

def init_weights(input_dimention, output_dimention):
	W = torch.randn(input_dimention, output_dimention) # C * C
	W.requires_grad = True
	return W


def train(X_encoding: torch.tensor, Y_label: torch.tensor, W: torch.tensor):
	for i in range(10000):
		idx = torch.randint(0, X_encoding.shape[0], (BATCH_SIZE,))
		X_batch = X_encoding[idx]
		Y_batch = Y_label[idx]

		logits = X_batch @ W # B * C
		loss = cross_entropy(logits, Y_batch)
		if i % 1000 == 0:
			print(loss.item())

		# back propogation
		loss.backward()

		# change parameters
		if i < 5000:
			lr = 0.5
		else:
			lr = 0.1
		with torch.no_grad():
			W -= lr * W.grad

		W.grad.zero_() #ERROR-4: W's gradients should not be set to zero before the first back propogation!


def softmax(logits: torch.tensor):
	exp = torch.exp(logits) # B * C
	sum_ = torch.sum(exp, dim=1, keepdim=True) # B * 1
	probs = exp / sum_ # B * C
	return probs

def cross_entropy(logits, Y_label):
	probs = softmax(logits) # B * C

	prediction = probs[torch.arange(len(Y_label)), Y_label] 
	# ERROR-1: torch.arange != torch.range; ERROR-2: forget to set the indexes of dimention 0
	
	loss = -torch.log(prediction).mean(dim=0) 
	# ERROR-3: mean instead of sum!
	return loss


def watch_W(W):
	# draw the hot map of W
	import plt
	with torch.no_grad():
		plt.imshow(softmax(W))
		plt.show()


def generate(W, encoder, decoder):
	x = '$'
	res = ""

	while True:
		x_gen = F.one_hot(torch.tensor([encoder[x]]), len(encoder)).float() # 1 * C
		logits = x_gen @ W # 1 * C
		probs = softmax(logits) # 1 * C
		# print(probs)

		# sample the next token based on the probs (the max prob)
		print(probs[0])
		m = torch.distributions.Categorical(probs[0]) # C
		sample = m.sample().item() # 1
		print(f"sample {sample}")

		# decode the next character
		x = decoder[sample]

		# reach the end
		if x == '$':
			break

		# append x
		res += x
	return res


def main():
	names = acquire_names()
	X_train, Y_train = form_training_set(names)
	encoder, decoder = gen_encoder_decoder(names)


	X_embedding, Y_label = encode(X_train, Y_train, encoder)

	W = init_weights(X_embedding.shape[1], X_embedding.shape[1])
	train(X_embedding, Y_label, W)

	watch_W(W)

	for _ in range(10):
		res = generate(W, encoder, decoder)
		print(res)

main()

# loss:
# 3.841094493865967
# 3.0871357917785645
# 2.242928981781006
# 2.4447386264801025
# 2.3591105937957764
# 2.6378555297851562
# 2.3850317001342773
# 2.4950034618377686
# 2.443566083908081
# 2.4574532508850098

# prediction:
# jahastatkshaile
# gisdrha
# bbuis
# ora
# aiesel
# n
# ama
# caereshiely
# gynarabele
# ah