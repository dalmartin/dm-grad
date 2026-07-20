from sympy.polys.specialpolys import w_polys
import torch
import torch.nn.functional as F

# Load the data
data = open("names.txt", "r").read().splitlines()
bigrams = {}

chars: set[str] = set()
for entry in data:
    for x in entry:
        chars.add(x)

charlist: list[str] = sorted(list(chars))
charlist.insert(0, '.')

cr: dict[str,  int] = {}
for i, ch in enumerate(charlist):
    cr[ch] = i

xs: list[int] = []
ys: list[int] = []

# For every data entry, get synced input and output integer (correspond to letter)
for entry in data[:1]:
    itr = '.' + entry + '.'
    for ch1, ch2 in zip(itr, itr[1:]):
        xs.append(cr[ch1])
        ys.append(cr[ch2])

# Convert to tensor
xts = torch.tensor(xs)
yts = torch.tensor(ys)

# Encode as a one hot encoding of the character (num classes = 27)
xenc = F.one_hot(xts, num_classes=27).float() # Needs to be cast to float because one_hot maintains datatypes

# initialize random weights
W = torch.randn((27, 27))
outputs = xenc @ W


