import torch # Pytorch :D
import matplotlib.pyplot as plt

# Get training data
data = open("names.txt", "r").read().splitlines()
bigrams = {}

chars: set[str] = set()
for entry in data:
    for x in entry:
        chars.add(x)

charlist: list[str] = sorted(list(chars))
charlist.append('<S>')
charlist.append('<E>')

cr: dict[str,  int] = {}
for i, ch in enumerate(charlist):
    cr[ch] = i

t = torch.zeros((27, 27), dtype=torch.int32)

for entry in data:
    for ch1, ch2 in zip(entry, entry[1:]):
        idx1: int = cr[ch1]
        idx2: int = cr[ch2]
        t[idx1, idx2] += 1

plt.imshow(t)
plt.show()
