# Logit lens: read out the next-letter guess after each block, from the weights in attention.js.
# Usage: python analysis/logit_lens.py attention.js text.txt
import re,json,base64,sys
import numpy as np
s=open(sys.argv[1],encoding="utf-8").read()
i=s.find('const MODEL = ')+len('const MODEL = ')
M,_=json.JSONDecoder().raw_decode(s,i)
b=base64.b64decode(M['b64']); P={}
for t in M['tensors']:
    n=int(np.prod(t['s']))
    if t['k']=='f32': P[t['n']]=np.frombuffer(b,np.float32,n,t['o']).reshape(t['s']); continue
    R,K=t['s']; sc=np.frombuffer(b,np.float32,R,t['o']); q=np.frombuffer(b,np.int8,n,t['o']+4*R).reshape(R,K)
    P[t['n']]=q.astype(np.float32)*sc[:,None]
C,L,H,V=M['C'],M['L'],M['H'],M['V']; hs=C//H
stoi={c:i for i,c in enumerate(M['chars'])}
def ln(x,w,b): mu=x.mean(-1,keepdims=True); v=((x-mu)**2).mean(-1,keepdims=True); return (x-mu)/np.sqrt(v+1e-5)*w+b
def gelu(u): return 0.5*u*(1+np.tanh(0.7978845608*(u+0.044715*u**3)))
def unembed(x):
    z=ln(x,P['lnf_w'],P['lnf_b'])@P['wte'].T; z-=z.max(-1,keepdims=True); p=np.exp(z); return p/p.sum(-1,keepdims=True)
def run(ids):
    T=len(ids); x=P['wte'][ids]+P['wpe'][:T]; outs=[unembed(x)]
    mask=np.triu(np.full((T,T),-1e30),1)
    for l in range(L):
        g=lambda n:P[f'h{l}.{n}']
        h=ln(x,g('ln1_w'),g('ln1_b')); qkv=h@g('qkv_w')+g('qkv_b')
        y=np.zeros((T,C),np.float32)
        for hh in range(H):
            q=qkv[:,hh*hs:(hh+1)*hs]; k=qkv[:,C+hh*hs:C+(hh+1)*hs]; v=qkv[:,2*C+hh*hs:2*C+(hh+1)*hs]
            a=q@k.T/np.sqrt(hs)+mask; a=np.exp(a-a.max(-1,keepdims=True)); a/=a.sum(-1,keepdims=True); y[:,hh*hs:(hh+1)*hs]=a@v
        x=x+y@g('proj_w')+g('proj_b')
        h2=ln(x,g('ln2_w'),g('ln2_b')); x=x+gelu(h2@g('fc_w')+g('fc_b'))@g('fc2_w')+g('fc2_b')
        outs.append(unembed(x))
    return outs
text=open(sys.argv[2],encoding="utf-8").read()
text=''.join(c for c in text if c in stoi)
T=M['T']; res=np.zeros((L+1,2)); n=0; ex=[]
for st in range(0,len(text)-T-1,T):
    ids=[stoi[c] for c in text[st:st+T]]; nxt=[stoi[c] for c in text[st+1:st+T+1]]
    outs=run(np.array(ids))
    for l,p in enumerate(outs):
        res[l,0]+=(p.argmax(-1)==nxt).sum(); res[l,1]+=np.log(p[np.arange(T),nxt]+1e-12).sum()
    n+=T
    if st==0:
        for l,p in enumerate(outs): ex.append(''.join(M['chars'][j] for j in p.argmax(-1)))
print(json.dumps({"chars_scored":n,"layers":[{"after":("embed" if l==0 else f"block {l}"),"top1":round(res[l,0]/n,3),"bits_per_char":round(-res[l,1]/n/np.log(2),3)} for l in range(L+1)],"input":text[:T],"greedy_by_layer":ex}, ensure_ascii=False, indent=1))
