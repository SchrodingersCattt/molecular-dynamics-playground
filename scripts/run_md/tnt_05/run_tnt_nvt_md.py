import time, sys, json, numpy as np
from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.verlet import VelocityVerlet
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from ase.neighborlist import natural_cutoffs, NeighborList
from scipy.sparse.csgraph import connected_components
from deepmd.calculator import DP
t0=time.time()
at=read("structures/ZZZMUC01.cif")
assert len(at)==168, len(at)
at=at.repeat((1,2,1))
write("structures/TNT_ZZZMUC01_1x2x1.extxyz",at)
write("structures/TNT_ZZZMUC01_1x2x1.vasp",at,format="vasp",direct=True,sort=True)
at.calc=DP(model="model/model.ckpt-1000000.pt")
def nmol(a):
    nl=NeighborList(natural_cutoffs(a,mult=1.15),self_interaction=False,bothways=True); nl.update(a)
    return connected_components(nl.get_connectivity_matrix(sparse=True))[0]
print("atoms",len(at),at.get_chemical_formula(),"cell",at.cell.lengths(),"nmol0",nmol(at),flush=True)
print("E0",at.get_potential_energy(),"Fmax0",np.abs(at.get_forces()).max(),flush=True)
np.random.seed(2026)
MaxwellBoltzmannDistribution(at,temperature_K=300); Stationary(at); ZeroRotation(at)
dt=0.5*units.fs
log=open("md.log","w"); log.write("# phase step time_fs Epot_eV Ekin_eV Etot_eV T_K nmol\n")
def run(dyn,nsteps,phase,dump_every,traj):
    for i in range(0,nsteps+1,dump_every):
        if i>0: dyn.run(dump_every)
        step=dyn.nsteps; ep=at.get_potential_energy(); ek=at.get_kinetic_energy()
        nm=nmol(at) if (i% (dump_every*20)==0) else -1
        log.write(f"{phase} {step} {step*0.5:.1f} {ep:.6f} {ek:.6f} {ep+ek:.6f} {at.get_temperature():.2f} {nm}\n"); log.flush()
        if traj: write(traj,at,append=True,format="extxyz")
# 1 ps NVT Langevin equilibration (friction 0.01/fs), 4 ps NVT production (gentle friction 0.002/fs) dump every 10 steps = 5 fs
eq=Langevin(at,dt,temperature_K=300,friction=0.01/units.fs)
run(eq,2000,"eq",20,None)
t1=time.time()
pr=Langevin(at,dt,temperature_K=300,friction=0.002/units.fs)
run(pr,8000,"prod",10,"tnt_nvt_300K_prod.extxyz")
t2=time.time()
print("nmol_final",nmol(at),"t_eq_s",t1-t0,"t_prod_s",t2-t1,flush=True)
json.dump({"t_total_s":t2-t0,"t_eq_s":t1-t0,"t_prod_s":t2-t1},open("timings.json","w"))
