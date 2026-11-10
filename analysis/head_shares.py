# Recompute the four README head shares from the shipped weights (read-only check).
# Usage: python analysis/head_shares.py attention.js text.txt analysis/logit_lens.py
import sys, numpy as np
ll = sys.argv[3]
src = open(ll, encoding='utf-8').read().split('text=open(sys.argv[2]')[0]
exec(src)
text = ''.join(c for c in open(sys.argv[2], encoding='utf-8').read() if c in stoi)
T = M['T']; VOW = set('aeiouAEIOU')
acc = {k: [0.0, 0] for k in ('L1H1 prev2', 'L2H4 last space/nl', 'L2H3 last nl', 'L3H2 vowels')}
for st in range(0, len(text) - T, T):
    w = text[st:st+T]; ids = np.array([stoi[c] for c in w]); TT = len(ids)
    x = P['wte'][ids] + P['wpe'][:TT]; mask = np.triu(np.full((TT, TT), -1e30), 1); att = {}
    for l in range(L):
        g = lambda n: P[f'h{l}.{n}']
        h = ln(x, g('ln1_w'), g('ln1_b')); qkv = h @ g('qkv_w') + g('qkv_b'); y = np.zeros((TT, C), np.float32)
        for hh in range(H):
            q = qkv[:, hh*hs:(hh+1)*hs]; k = qkv[:, C+hh*hs:C+(hh+1)*hs]; v = qkv[:, 2*C+hh*hs:2*C+(hh+1)*hs]
            a = q @ k.T / np.sqrt(hs) + mask; a = np.exp(a - a.max(-1, keepdims=True)); a /= a.sum(-1, keepdims=True)
            att[(l, hh)] = a; y[:, hh*hs:(hh+1)*hs] = a @ v
        x = x + y @ g('proj_w') + g('proj_b'); h2 = ln(x, g('ln2_w'), g('ln2_b')); x = x + gelu(h2 @ g('fc_w') + g('fc_b')) @ g('fc2_w') + g('fc2_b')
    for t in range(2, TT):
        a = att[(0, 0)][t]; acc['L1H1 prev2'][0] += a[t-1] + a[t-2]; acc['L1H1 prev2'][1] += 1
        sp = [j for j in range(t+1) if w[j] in ' \n']
        if sp: acc['L2H4 last space/nl'][0] += att[(1, 3)][t][sp[-1]]; acc['L2H4 last space/nl'][1] += 1
        nl = [j for j in range(t+1) if w[j] == '\n']
        if nl: acc['L2H3 last nl'][0] += att[(1, 2)][t][nl[-1]]; acc['L2H3 last nl'][1] += 1
        vw = [j for j in range(t+1) if w[j] in VOW]
        if vw: acc['L3H2 vowels'][0] += att[(2, 1)][t][vw].sum(); acc['L3H2 vowels'][1] += 1
for k, (s_, n) in acc.items(): print(f'{k}: {s_/n*100:.1f}% (n={n})')
