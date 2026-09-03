"""
Finite-shot neural decoding for superdense coding.

Reproduces the superdense-coding results of the manuscript
"Neural-Network Error Mitigation for Quantum Teleportation and Superdense Coding"
(Table 1 and Figure 3).

Compares three decoders vs number of measurement shots:
  1. Majority vote over finite-shot counts (weak fixed rule).
  2. Channel-aware maximum-likelihood (ML) decoder using the known measurement
     model (non-neural, training-free) -- the fair baseline.
  3. Learned decoder (small feed-forward network on empirical frequencies).

Noise model: biased Pauli channel (p_z in [0.05,0.35], p_x in [0,0.10] per
transmission) followed by the fixed asymmetric readout confusion matrix Cm.
Results are means over independent seeds with 95% confidence intervals.

Key finding: the learned decoder beats majority vote but only MATCHES the ML
decoder -- it approximates the optimal decoder rather than exceeding it.

Run:  python superdense_finite_shot.py
For tighter intervals, increase SEEDS at the bottom of the file.
"""

import numpy as np, torch, torch.nn as nn

I=np.eye(2,dtype=complex); X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex); Z=np.array([[1,0],[0,-1]],dtype=complex)
H=(1/np.sqrt(2))*np.array([[1,1],[1,-1]],dtype=complex)
CNOT=np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]],dtype=complex)
bell=(1/np.sqrt(2))*np.array([1,0,0,1],dtype=complex)
gates={0:I,1:X,2:Z,3:X@Z}

Cm=np.array([[0.92,0.05,0.05,0.00],
             [0.10,0.85,0.00,0.05],
             [0.10,0.00,0.85,0.05],
             [0.00,0.08,0.08,0.84]])

def outcome_probs(msg, pz, px):
    g=gates[msg]; enc=np.kron(g,I)@bell; rho=np.outer(enc,enc.conj())
    K=[np.sqrt(1-pz-px)*I, np.sqrt(pz)*Z, np.sqrt(px)*X]
    rho=sum(np.kron(k,I)@rho@np.kron(k,I).conj().T for k in K)
    U=np.kron(H,I)@CNOT; rho=U@rho@U.conj().T
    pr=np.clip(np.real(np.diag(rho)),0,None)
    pr=Cm.T@pr; pr=np.clip(pr,0,None); pr/=pr.sum()
    return pr

def make_dataset(N, shots):
    msgs=np.random.randint(0,4,N)
    pz=np.random.uniform(0.05,0.35,N); px=np.random.uniform(0.0,0.10,N)
    counts=np.zeros((N,4),np.float32)
    probs=np.zeros((N,4))
    for i in range(N):
        pr=outcome_probs(msgs[i],pz[i],px[i]); probs[i]=pr
        draws=np.random.choice(4,size=shots,p=pr)
        for d in draws: counts[i,d]+=1
    return msgs, counts/shots, counts, pz, px

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(4,128),nn.ReLU(),
                               nn.Linear(128,128),nn.ReLU(),nn.Linear(128,4))
    def forward(self,x): return self.net(x)

# Precompute expected outcome distribution for each message at a representative
# mid-range noise (pz,px) for a channel-aware ML decoder that does NOT know per-sample p.
def ml_templates():
    pz,px=0.20,0.05  # mid-range assumption
    return np.stack([outcome_probs(m,pz,px) for m in range(4)])  # 4x4

def one_run(shots, seed, N=9000, epochs=1000):
    np.random.seed(seed); torch.manual_seed(seed)
    msgs,freqs,counts,pz,px=make_dataset(N,shots)
    ntr=7000
    Xtr,Ytr=torch.tensor(freqs[:ntr]),torch.tensor(msgs[:ntr])
    Xte=freqs[ntr:]; Yte=msgs[ntr:]; counts_te=counts[ntr:]

    # baseline 1: majority vote
    maj=np.argmax(counts_te,1); acc_maj=(maj==Yte).mean()

    # baseline 2: channel-aware ML decoder (multinomial likelihood vs templates)
    T=ml_templates()                      # 4 x 4 expected distributions
    logT=np.log(np.clip(T,1e-9,None))     # 4 x 4
    # loglik for each test sample and each hypothesis message = counts . logT[msg]
    ll=counts_te@logT.T                   # (Nte x 4)
    mldec=np.argmax(ll,1); acc_ml=(mldec==Yte).mean()

    # neural decoder
    m=Decoder(); opt=torch.optim.Adam(m.parameters(),2e-3); lf=nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xtr),Ytr).backward(); opt.step()
    with torch.no_grad(): pred=m(torch.tensor(Xte)).argmax(1).numpy()
    acc_nn=(pred==Yte).mean()
    return acc_maj, acc_ml, acc_nn

def summary(vals):
    a=np.array(vals); m=a.mean(); half=1.96*a.std(ddof=1)/np.sqrt(len(a))
    return m, half

if __name__=="__main__":
    SEEDS=[4,5,6]
    for s in [1,5,10,20,50,100]:
        mj,ml,nnv=[],[],[]
        for sd in SEEDS:
            am,al,an=one_run(s,sd)
            mj.append(am);ml.append(al);nnv.append(an)
        mm,mh=summary(mj); lm,lh=summary(ml); nm,nh=summary(nnv)
        print(f"{s:4d} | maj {mm:.3f}+/-{mh:.3f} | ML {lm:.3f}+/-{lh:.3f} | NN {nm:.3f}+/-{nh:.3f}")
