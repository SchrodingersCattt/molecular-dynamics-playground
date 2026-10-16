import sys, numpy as np, time
sys.path.insert(0, "/aisi-nas/guomingyu/personal/mlip-playground/261013_tnt_uks_bs")
import run_tnt_uks as R
d = np.load("scf_in.npz"); el=[str(x) for x in d["elements"]]; P=d["positions"]; axes=(d["gx"],d["gy"],d["gz"])
ck = np.load("/aisi-nas/guomingyu/personal/mlip-playground/261013_tnt_uks_bs/scf_checkpoint.npz")
for ion in [int(x) for x in sys.argv[1:]]:
    t=time.time(); dm0 = ck["dms"][ion-1]
    seq=[np.asarray(dm0)]; conv=False; k=0
    while not conv and k < 40:
        k += 1
        mf = R.make_uks(R.make_mol(el, P[ion]), 0.0); mf.conv_tol=1e-8; mf.max_cycle=k
        mf.kernel(dm0=(np.asarray(dm0[0]), np.asarray(dm0[1])))
        seq.append(R.to_np(mf.make_rdm1())); conv=bool(mf.converged)
    rho = np.stack([R.eval_density(el, P[ion], dm, axes)[0] for dm in seq])
    np.savez_compressed(f"scf_iterates_{ion:02d}.npz", rho=rho)
    print(ion, "cycles", k, "rel", np.round([float(np.abs(r-rho[-1]).max()/rho[-1].max()) for r in rho],5).tolist(), f"{time.time()-t:.0f}s", flush=True)
