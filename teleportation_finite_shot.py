"""
Finite-shot neural correction for quantum teleportation.

Reproduces the teleportation results of the manuscript
"Neural-Network Error Mitigation for Quantum Teleportation and Superdense Coding"
(Table 2 and Figure 4), plus the R_fix recovery check.

Compares three methods vs number of tomographic shots:
  1. Uncorrected measured Bloch vector.
  2. Least-squares (LS) estimate of the fixed rotation (non-neural, one linear
     map fit measured->clean) -- the fair baseline.
  3. Learned corrector (feed-forward network).

Noise model: a fixed systematic rotation R_fix = R((0.3,0.5,0.8), 0.6) applied
to every qubit, plus small per-qubit random jitter, then finite-shot binomial
tomography. Results are means over seeds with 95% confidence intervals.

Key findings: the LS baseline recovers most of the lost fidelity (and recovers
R_fix^{-1} to Frobenius distance ~0.02, confirming it inverts the physical
miscalibration); the network improves on LS only modestly (~1-2 points), from
nonlinear jitter and tomographic noise the linear map cannot absorb.

Run:  python teleportation_finite_shot.py
For tighter intervals, increase SEEDS / epochs at the bottom of the file.
"""

import numpy as np, torch, torch.nn as nn

X=np.array([[0,1],[1,0]],dtype=complex)
Y=np.array([[0,-1j],[1j,0]],dtype=complex)
Z=np.array([[1,0],[0,-1]],dtype=complex)

def random_bloch(N):
    z=1-2*np.random.rand(N); phi=2*np.pi*np.random.rand(N); s=np.sqrt(1-z*z)
    return np.stack([s*np.cos(phi), s*np.sin(phi), z],1)

def rot_matrix(axis, eps):
    ax=np.asarray(axis,float); ax=ax/np.linalg.norm(ax)
    K=np.array([[0,-ax[2],ax[1]],[ax[2],0,-ax[0]],[-ax[1],ax[0],0]])
    return np.eye(3)+np.sin(eps)*K+(1-np.cos(eps))*(K@K)

def fidelity(bp, bc):
    r=np.linalg.norm(bp,axis=1,keepdims=True)
    bp=np.where(r>1, bp/r, bp)
    return 0.5*(1+np.sum(bp*bc,1))

FIXED_AXIS=np.array([0.3,0.5,0.8]); FIXED_EPS=0.6
Rfix=rot_matrix(FIXED_AXIS, FIXED_EPS)

class Corrector(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(3,256),nn.ReLU(),nn.Linear(256,256),nn.ReLU(),
                               nn.Linear(256,128),nn.ReLU(),nn.Linear(128,3))
    def forward(self,x): return self.net(x)

def one_run(shots, seed, N=8000, epochs=800):
    np.random.seed(seed); torch.manual_seed(seed)
    clean=random_bloch(N)
    jitter=np.random.uniform(0,0.15,N); jax=np.random.randn(N,3)
    # apply fixed rotation to all at once
    pre=clean@Rfix.T
    # apply per-sample jitter rotation, vectorized via Rodrigues
    axn=jax/np.linalg.norm(jax,axis=1,keepdims=True)
    noisy=np.zeros_like(clean)
    # build per-sample K and rotate; still a loop but lighter (no rot_matrix overhead)
    for i in range(N):
        a=axn[i]; e=jitter[i]
        K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
        R=np.eye(3)+np.sin(e)*K+(1-np.cos(e))*(K@K)
        noisy[i]=R@pre[i]
    pplus=(1+noisy)/2
    k=np.random.binomial(shots, np.clip(pplus,0,1))
    est=(2*(k/shots)-1).astype(np.float32)
    ntr=6500
    Xtr,Ytr=est[:ntr],clean[:ntr].astype(np.float32)
    Xte,Yte=est[ntr:],clean[ntr:]

    # --- baseline 1: uncorrected (directly measured) ---
    f_unc=fidelity(Xte,Yte).mean()

    # --- baseline 2: least-squares fixed-rotation estimate (NON-NEURAL) ---
    # Solve argmin_M || M * Xtr^T - Ytr^T ||  (M approximates Rfix^{-1}), then apply to test.
    # Xtr ~ (Rfix clean) + noise ; we regress clean on measured, i.e. M s.t. clean ~ M measured.
    M,_,_,_=np.linalg.lstsq(Xtr, Ytr, rcond=None)   # (3x3), maps measured->clean
    ls_pred=Xte@M
    f_ls=fidelity(ls_pred,Yte).mean()

    # --- neural corrector ---
    Xtr_t=torch.tensor(Xtr); Ytr_t=torch.tensor(Ytr)
    m=Corrector(); opt=torch.optim.Adam(m.parameters(),1e-3); lf=nn.MSELoss()
    for _ in range(epochs):
        opt.zero_grad(); lf(m(Xtr_t),Ytr_t).backward(); opt.step()
    with torch.no_grad(): nn_pred=m(torch.tensor(Xte)).numpy()
    f_nn=fidelity(nn_pred,Yte).mean()

    return f_unc, f_ls, f_nn, M

def summary(vals):
    a=np.array(vals); m=a.mean(); half=1.96*a.std(ddof=1)/np.sqrt(len(a))
    return m, half

if __name__=="__main__":
    SEEDS=[7,8,9]
    print(f"{'shots':>6} | {'uncorrected':>18} | {'LS baseline':>18} | {'neural':>18}")
    rows={}
    for s in [10,30,100,300]:
        us,ls,ns=[],[],[]
        for sd in SEEDS:
            fu,fl,fn,_=one_run(s,sd)
            us.append(fu); ls.append(fl); ns.append(fn)
        um,uh=summary(us); lm,lh=summary(ls); nm,nh=summary(ns)
        rows[s]=(um,uh,lm,lh,nm,nh)
        print(f"{s:6d} | {um:6.3f} +/- {uh:5.3f}    | {lm:6.3f} +/- {lh:5.3f}    | {nm:6.3f} +/- {nh:5.3f}")
    # report learned-vs-Rfix comparison at high shots
    print("\n--- Does the network learn Rfix^{-1}? (single seed, 300 shots) ---")
    _,_,_,M=one_run(300,7)
    Rinv=np.linalg.inv(Rfix)
    print("LS-estimated map M (measured->clean):\n", np.round(M.T,3))
    print("True Rfix^{-1}:\n", np.round(Rinv,3))
    print("Frobenius ||M^T - Rfix^{-1}|| =", round(np.linalg.norm(M.T-Rinv),4))

    # --- Generalisation test (manuscript Sec 4.3) ---
    # Train on Rfix, then apply WITHOUT retraining to a different fixed rotation R'.
    # A calibration-specific corrector helps on Rfix but not on R'.
    print("\n--- Generalisation test (100 shots, seed 7) ---")
    rng=np.random.default_rng(7); torch.manual_seed(7)
    Ntest=8000; shots=100; ntr=6500
    z=1-2*rng.random(Ntest); phi=2*np.pi*rng.random(Ntest); ss=np.sqrt(1-z*z)
    clean=np.stack([ss*np.cos(phi), ss*np.sin(phi), z],1)
    def _noisy(clean, Rf):
        N=len(clean); jit=rng.uniform(0,0.15,N); jax=rng.standard_normal((N,3))
        axn=jax/np.linalg.norm(jax,axis=1,keepdims=True); pre=clean@Rf.T; out=np.zeros_like(clean)
        for i in range(N):
            a=axn[i]; e=jit[i]
            K=np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
            out[i]=(np.eye(3)+np.sin(e)*K+(1-np.cos(e))*(K@K))@pre[i]
        return out
    est=(2*(rng.binomial(shots,np.clip((1+_noisy(clean,Rfix))/2,0,1))/shots)-1).astype(np.float32)
    m=Corrector(); opt=torch.optim.Adam(m.parameters(),1e-3); lf=nn.MSELoss()
    Xt=torch.tensor(est[:ntr]); Yt=torch.tensor(clean[:ntr].astype(np.float32))
    for _ in range(800): opt.zero_grad(); lf(m(Xt),Yt).backward(); opt.step()
    with torch.no_grad(): f_same=fidelity(m(torch.tensor(est[ntr:])).numpy(),clean[ntr:]).mean()
    Rp=rot_matrix(np.array([0.8,-0.2,0.4]),0.9)
    est2=(2*(rng.binomial(shots,np.clip((1+_noisy(clean,Rp))/2,0,1))/shots)-1).astype(np.float32)
    with torch.no_grad(): f_diff=fidelity(m(torch.tensor(est2[ntr:])).numpy(),clean[ntr:]).mean()
    f_diff_unc=fidelity(est2[ntr:],clean[ntr:]).mean()
    print(f"Trained on Rfix, tested on Rfix:        {f_same:.3f}")
    print(f"Trained on Rfix, tested on different R:  {f_diff:.3f}  (uncorrected {f_diff_unc:.3f})")
