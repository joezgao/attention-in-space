# Zero-ablate each attention head of the Attention, in space GPT and measure the change in bits/char.
# Usage: python analysis/head_ablation.py attention.js text.txt
import sys, json
import numpy as np
sys.argv_saved = list(sys.argv)
src = open(__file__.replace('head_ablation.py', 'logit_lens.py'), encoding='utf-8').read()
src = src.split('text=open(sys.argv[2]')[0]          # reuse the weight decoding and helpers
exec(src)
def run_ablate(ids, kill=None):
    T=len(ids); x=P['wte'][ids]+P['wpe'][:T]; mask=np.triu(np.full((T,T),-1e30),1)
    for l in range(L):
        g=lambda n:P[f'h{l}.{n}']
        h=ln(x,g('ln1_w'),g('ln1_b')); qkv=h@g('qkv_w')+g('qkv_b'); y=np.zeros((T,C),np.float32)
        for hh in range(H):
            if kill==(l,hh): continue
            q=qkv[:,hh*hs:(hh+1)*hs]; k=qkv[:,C+hh*hs:C+(hh+1)*hs]; v=qkv[:,2*C+hh*hs:2*C+(hh+1)*hs]
            a=q@k.T/np.sqrt(hs)+mask; a=np.exp(a-a.max(-1,keepdims=True)); a/=a.sum(-1,keepdims=True); y[:,hh*hs:(hh+1)*hs]=a@v
        x=x+y@g('proj_w')+g('proj_b'); h2=ln(x,g('ln2_w'),g('ln2_b')); x=x+gelu(h2@g('fc_w')+g('fc_b'))@g('fc2_w')+g('fc2_b')
    return unembed(x)
text=''.join(c for c in open(sys.argv[2],encoding='utf-8').read() if c in stoi); T=M['T']
segs=[(np.array([stoi[c] for c in text[s:s+T]]),np.array([stoi[c] for c in text[s+1:s+T+1]])) for s in range(0,len(text)-T-1,T)]
def bpc(kill=None):
    tot=0;n=0
    for ids,nx in segs:
        p=run_ablate(ids,kill); tot+=-np.log2(p[np.arange(T),nx]+1e-12).sum(); n+=T
    return tot/n
base=bpc(); res={f'L{l+1}H{h+1}':round(float(bpc((l,h))-base),3) for l in range(L) for h in range(H)}
print(json.dumps({'base_bits_per_char':round(float(base),3),'delta_bits_when_head_removed':res},indent=1))
