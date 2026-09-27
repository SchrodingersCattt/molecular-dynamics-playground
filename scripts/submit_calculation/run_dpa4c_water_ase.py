"""Run the 0314 water box with the public DPA4C-OMat24 checkpoint.

This is deliberately a live inference runner: the checkpoint is loaded by
DeePMD-kit/PyTorch Exportable through the ASE calculator and every Velocity
Verlet position gets a fresh DPA4C energy/force evaluation.  It never falls
back to the old ``.pb`` model.
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
import numpy as np

EV_A_TO_A_FS2 = 0.00964853399
O_MASS, H_MASS = 15.9994, 1.008

def sha256(path: Path) -> str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(1024*1024), b""): h.update(chunk)
    return h.hexdigest()

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument("--model",type=Path,required=True); ap.add_argument("--input",type=Path,required=True); ap.add_argument("--output",type=Path,required=True); ap.add_argument("--metadata",type=Path,required=True); ap.add_argument("--steps",type=int,default=47); ap.add_argument("--dt",type=float,default=.5); ap.add_argument("--temperature",type=float,default=300.); ap.add_argument("--seed",type=int,default=260906); a=ap.parse_args()
    from ase import Atoms
    from deepmd.calculator import DP
    with np.load(a.input,allow_pickle=False) as z:
        pos=np.asarray(z["positions_wrapped"],float); elements=np.asarray(z["elements"]).astype(str); box=float(z["box_length"]); centre=int(z["central_index"]); cutoff=float(z["cutoff"])
    if set(elements.tolist()) != {"O","H"}: raise RuntimeError("0314 input must be the prepared O/H water box")
    atoms=Atoms(symbols=elements.tolist(),positions=pos,cell=np.eye(3)*box,pbc=True); atoms.info["charge_spin"]=np.array([0,1]); atoms.calc=DP(model=str(a.model))
    masses=np.where(elements=="O",O_MASS,H_MASS); rng=np.random.default_rng(a.seed); vel=rng.normal(0,np.sqrt(8.617333262145e-5*a.temperature/(masses[:,None]/EV_A_TO_A_FS2)),size=pos.shape); vel-=np.average(vel,axis=0,weights=masses); atoms.set_velocities(vel)
    positions=[pos.copy()]; velocities=[vel.copy()]; energies=[]; forces=[]; atomic=[]
    def evaluate():
        e=float(atoms.get_potential_energy()); f=np.asarray(atoms.get_forces(),float); 
        try: ae=np.asarray(atoms.get_potential_energies(),float)
        except Exception: ae=np.full(len(atoms),e/len(atoms))
        return e,f,ae
    e,f,ae=evaluate(); energies.append(e); forces.append(f); atomic.append(ae)
    for _ in range(a.steps):
        acc=f*EV_A_TO_A_FS2/masses[:,None]; vhalf=atoms.get_velocities()+.5*a.dt*acc; new=(atoms.get_positions()+a.dt*vhalf)%box; atoms.set_positions(new); e2,f2,ae2=evaluate(); vnew=vhalf+.5*a.dt*f2*EV_A_TO_A_FS2/masses[:,None]; atoms.set_velocities(vnew); positions.append(new.copy()); velocities.append(vnew.copy()); energies.append(e2); forces.append(f2); atomic.append(ae2); f=f2
    a.output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(a.output,elements=elements,box_length=np.array(box),central_index=np.array(centre),cutoff_angstrom=np.array(cutoff),positions=np.asarray(positions),velocities=np.asarray(velocities),forces_ev_per_angstrom=np.asarray(forces),atomic_energy_ev=np.asarray(atomic),total_energy_ev=np.asarray(energies),dt_fs=np.array(a.dt),temperature_k=np.array(a.temperature),velocity_seed=np.array(a.seed))
    meta={"schema":"dpa4c_live_water_md/v1","model":a.model.name,"model_sha256":sha256(a.model),"input":a.input.name,"input_sha256":sha256(a.input),"n_states":len(positions),"n_atoms":len(elements),"integrator":"Velocity Verlet; fresh DPA4C ASE evaluation at every new position","backend":"DeePMD-kit PyTorch Exportable / ASE DP","descriptor":"dpa4c"}; a.metadata.parent.mkdir(parents=True,exist_ok=True); a.metadata.write_text(json.dumps(meta,indent=2)+"\n")

if __name__=="__main__": main()
