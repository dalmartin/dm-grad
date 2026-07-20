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
charlist.insert(0, '.')

cr: dict[str,  int] = {}
for i, ch in enumerate(charlist):
    cr[ch] = i

t = torch.zeros((27, 27), dtype=torch.int32)

for entry in data:
    itr = '.' + entry + '.'
    for ch1, ch2 in zip(itr, itr[1:]):
        idx1: int = cr[ch1]
        idx2: int = cr[ch2]
        t[idx1, idx2] += 1


# t contains "trained" probabilities for bigrams
g = torch.Generator()
idxch = {v: k for k, v in cr.items()}


P = t.float()
P /= P.sum(1, keepdims=True)

for n in range(20):
    f = 0
    out = []
    while True:
        # p = t[f].float()
        # p = p / p.sum()
        
        p = P[f]

        f = torch.multinomial(p, num_samples=1, replacement=True, generator=g).item()
        out.append(idxch[f])
        if f == 0:
            break
    
    print(''.join(out))



# Average negative log liklihood (evaluate the model)
log_likelihood = 0.0
n = 0

for entry in data:
    itr = '.' + entry + '.'
    for ch1, ch2 in zip(itr, itr[1:]):
        idx1: int = cr[ch1]
        idx2: int = cr[ch2]
        
        prob = P[idx1, idx2]
        log_likelihood += torch.log(prob)
        n += 1

nll = (log_likelihood) / -n
print(f'Negative Log Likelihood: {nll}')
