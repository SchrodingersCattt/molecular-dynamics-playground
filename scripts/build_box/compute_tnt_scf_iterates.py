import sys, numpy as np, time
sys.path.insert(0, "/aisi-nas/guomingyu/personal/mlip-playground/261013_tnt_uks_bs")
import run_tnt_uks as R
d = np.load("scf_in.npz"); el=[str(x) for x in d["elements"]]; P=d["positions"]; axes=(d["gx"],d["gy"],d["gz"])
ck = np.load("/aisi-nas/guomingyu/personal/mlip-playground/261013_tnt_uks_bs/scf_checkpoint.npz")
out = {}
for ion in (0, 1):
    t=time.time()
    dm0 = R.fragment_guess(el, P[ion]) if ion == 0 else ck["dms"][0]
    seq=[np.asarray(dm0)]; energies=[]; conv=False; k=0
    while not conv and k < 40:
        k += 1
        mol = R.make_mol(el, P[ion]); mf = R.make_uks(mol, 0.0); mf.conv_tol=1e-8; mf.max_cycle=k
        e = mf.kernel(dm0=(np.asarray(dm0[0]), np.asarray(dm0[1])))
        seq.append(R.to_np(mf.make_rdm1())); energies.append(float(R.to_np(e))); conv=bool(mf.converged)
    rho = np.stack([R.eval_density(el, P[ion], dm, axes)[0] for dm in seq])
    rel = [float(np.abs(r-rho[-1]).max()/rho[-1].max()) for r in rho]
    ddm = [float(np.abs(seq[i+1]-seq[i]).max()) for i in range(len(seq)-1)]
    print(ion, "cycles", k, "conv", conv, "dE", np.round(np.array(energies)-energies[-1],8).tolist(), "max|ddm|", np.round(ddm,8).tolist(), "rel_rho_to_final", np.round(rel,5).tolist(), f"{time.time()-t:.0f}s", flush=True)
    out[f"rho_{ion}"]=rho; out[f"ddm_{ion}"]=np.asarray(ddm); out[f"energy_{ion}"]=np.asarray(energies)
np.savez_compressed("scf_iterates.npz", **out)
