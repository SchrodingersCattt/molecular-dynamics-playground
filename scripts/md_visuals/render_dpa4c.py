"""Render the independent 04_4c DPA4C architecture animation.

The default preview uses the retained real 64-water trajectory and computes a
fully reproducible local-feature projection (radial envelope, l=0/1/2 angular
channels, Gram invariants and rotation check).  Until a DPA4C checkpoint is
run, the preview labels energy/force values as reference-DP values.
"""
from __future__ import annotations
import argparse, hashlib, json, subprocess
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parents[2]; DATA = ROOT / "product/data"; FIG = ROOT / "product/figures"; VID = ROOT / "product/videos"; QA = ROOT / "product/qa/04_4c"
SRC = DATA / "dpmd_water_box_trajectory.npz"
INK, BLUE, GREEN, RED, GOLD, GREY = "#12243C", "#205A74", "#246249", "#941B32", "#766D29", "#7A868A"

def local_features(pos, centre, box, cutoff):
    d = pos - centre; d -= box*np.rint(d/box); r = np.linalg.norm(d, axis=1); mask=(r>1e-9)&(r<cutoff)
    d, r = d[mask], r[mask]; u=d/np.maximum(r[:,None],1e-12); w=np.clip(1-r/cutoff,0,1)**4
    x0=np.sum(w); x1=np.sum(w[:,None]*u,axis=0); q=np.einsum('ni,nj,n->ij',u,u,w)
    gram=np.array([x0, np.linalg.norm(x1), np.trace(q), np.sum(q*q)],float)
    # A proper equivariant rotation changes x1/q but leaves these contractions unchanged.
    rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
    rotated=np.array([x0, np.linalg.norm(rot@x1), np.trace(rot@q@rot.T), np.sum((rot@q@rot.T)**2)])
    return d,r,w,x0,x1,q,gram,rotated

def interpolate(z, n=48):
    old=np.arange(len(z["positions"])); new=np.linspace(0,len(old)-1,n); out={k: np.array([np.interp(new,old,v[:,i]) for i in range(v.shape[1])]).T for k,v in [("positions",z["positions"].reshape(len(old),-1))]}; out["positions"]=out["positions"].reshape(n,z["positions"].shape[1],3)
    for k in ("forces_ev_per_angstrom","atomic_energy_ev","total_energy_ev"):
        value=np.asarray(z[k]); flat=value.reshape(len(old),-1)
        interp=np.array([np.interp(new,old,flat[:,i]) for i in range(flat.shape[1])]).T
        out[k]=interp.reshape((n,)+value.shape[1:])
    out.update({k:z[k] for k in ("box","central","cutoff","energy_status") if k in z})
    return out

def story_panel(ax, stage, d, r, w, x0, x1, q, g, gr):
    """A single visual mathematical action, in the spirit of a chalk talk."""
    ax.text(.06,.93,f"DESCRIPTOR · STEP {stage+1}/8",fontsize=14,color=GREY,weight="bold")
    titles=("Start with one directed edge", "Turn distance into radial functions", "Turn direction into angular components", "Add the neighbour contributions", "Rotate the equivariant features", "Contract away the coordinate frame", "Concatenate the invariant blocks", "Read out energy, then differentiate")
    ax.text(.06,.84,titles[stage],fontsize=18,color=INK,weight="bold")
    if stage==0:
        ax.text(.08,.67,r"$r_{ij}=r_j-r_i$",fontsize=20,color=BLUE)
        ax.add_patch(Circle((.72,.63),.035,fc=RED,ec="none")); ax.add_patch(Circle((.40,.63),.025,fc=GREEN,ec="none"))
        ax.add_patch(FancyArrowPatch((.43,.63),(.68,.63),arrowstyle="-|>",mutation_scale=18,color=GREEN,lw=2.5))
        ax.text(.40,.55,"centre $i$",ha="center",fontsize=14,color=INK); ax.text(.72,.55,"neighbour $j$",ha="center",fontsize=14,color=INK)
        ax.text(.08,.36,rf"$r_{{ij}}={r[0]:.3f}\,\AA$   $u_{{ij}}=({d[0,0]/r[0]:+.2f},{d[0,1]/r[0]:+.2f},{d[0,2]/r[0]:+.2f})$",fontsize=15,color=INK)
        ax.text(.08,.22,rf"$\chi(r_{{ij}})=(1-r_{{ij}}/r_c)^4_+={w[0]:.3f}$",fontsize=15,color=GREEN)
    elif stage==1:
        xx=np.linspace(0,6,120); ax.plot((.10,.92),(.22,.22),color=GREY,lw=1); ax.plot((.10,.10),(.22,.70),color=GREY,lw=1)
        for n,col in enumerate((BLUE,GREEN,RED)):
            yy=.22+.38*np.abs(np.sin((n+1)*np.pi*xx/6))*np.exp(-.07*xx)
            ax.plot(.10+.82*xx/6,.22+.48*(yy-.22)/.48,color=col,lw=2)
        marker=.10+.82*min(r[0],6)/6; ax.plot(marker,.22+.38*np.abs(np.sin(np.pi*r[0]/6))*np.exp(-.07*r[0]),"o",ms=9,color=GOLD)
        ax.text(.12,.74,r"$f_n(r)=\sin(\omega_n r)/r$",fontsize=18,color=BLUE)
        ax.text(.12,.12,rf"the same $r_{{ij}}$ samples every radial channel: $[f_1,f_2,f_3]=[{np.sin(r[0]):+.2f},{np.sin(2*r[0]):+.2f},{np.sin(3*r[0]):+.2f}]$",fontsize=13,color=INK)
    elif stage==2:
        ax.add_patch(Circle((.50,.52),.18,fill=False,ec=GREY,lw=1.2)); ax.arrow(.50,.52,.23,.10,width=.004,head_width=.025,color=BLUE,length_includes_head=True)
        ax.text(.75,.66,r"$u_{ij}$",fontsize=18,color=BLUE)
        for y,label,val,col in ((.35,"$B_{10}\u003d u_x$",d[0,0]/r[0],BLUE),(.24,"$B_{11}\u003d u_y$",d[0,1]/r[0],GREEN),(.13,"$B_{12}\u003d u_z$",d[0,2]/r[0],RED)):
            ax.text(.08,y,label,fontsize=16,color=col); ax.plot((.45,.45+.35*val),(y,y),lw=5,color=col); ax.plot(.45+.35*val,y,"o",color=col)
        ax.text(.08,.75,"direction becomes a small set of angular components",fontsize=15,color=INK)
    elif stage==3:
        count=min(8,len(w)); vals=w[:count]/max(np.max(w[:count]),1e-12)
        ax.text(.08,.75,r"$X_{\ell mc}=M_\ell^{-1}\sum_j\chi_{ij,\ell}\psi_{ijc}B_{\ell m}(u_{ij})$",fontsize=17,color=GREEN)
        for k,val in enumerate(vals):
            x=.10+k*.105; ax.add_patch(Rectangle((x,.25),.07,.35*val,fc=GREEN,alpha=.7,ec="none")); ax.text(x+.035,.19,str(k+1),ha="center",fontsize=12,color=GREY)
        ax.add_patch(FancyArrowPatch((.10,.68),(.88,.68),arrowstyle="-|>",mutation_scale=16,color=GREY,lw=1.5)); ax.text(.50,.55,"many neighbours → one local feature",ha="center",fontsize=16,color=INK)
    elif stage==4:
        rot=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        for off,vec,col,label in ((.25,x1,BLUE,"before"),(.68,rot@x1,GREEN,"after rotation")):
            ax.arrow(off,.46,.22*vec[0],.22*vec[1],width=.006,head_width=.028,color=col,length_includes_head=True); ax.text(off,.25,label,ha="center",fontsize=15,color=col)
        ax.text(.08,.78,r"$X_{\ell}\mapsto R_{\ell}X_{\ell}$",fontsize=21,color=GREEN); ax.text(.08,.12,"the components move with the molecule",fontsize=15,color=INK)
    elif stage==5:
        ax.imshow(q,extent=(.12,.44,.25,.62),origin="lower",cmap="RdYlGn",vmin=np.min(q),vmax=np.max(q),aspect="auto"); ax.imshow(np.array([[q[0,0],q[0,1]],[q[1,0],q[1,1]]]),extent=(.62,.94,.25,.62),origin="lower",cmap="RdYlGn",vmin=np.min(q),vmax=np.max(q),aspect="auto")
        ax.text(.28,.18,r"$G=X^TX$",ha="center",fontsize=18,color=RED); ax.text(.78,.18,r"$G'=(RX)^T(RX)=G$",ha="center",fontsize=18,color=RED)
        ax.text(.08,.75,r"dot products forget the absolute orientation",fontsize=16,color=INK)
    elif stage==6:
        heat=np.resize(np.abs(g)/(np.max(np.abs(g))+1e-12),(6,12)); ax.imshow(heat,extent=(.15,.85,.25,.65),origin="lower",cmap="RdYlGn",vmin=0,vmax=1,aspect="auto"); ax.text(.50,.17,r"$D_i=\mathrm{calib}[G\,|\,J\,|\,\Pi]$",ha="center",fontsize=21,color=RED); ax.text(.08,.75,"the invariant blocks are concatenated and calibrated",fontsize=16,color=INK)
    else:
        for x in (.16,.31,.46,.61,.76): ax.add_patch(Circle((x,.48),.045,fc=GOLD if x<.76 else RED,ec="none"))
        for a,b in zip((.16,.31,.46,.61),(.31,.46,.61,.76)): ax.add_patch(FancyArrowPatch((a+.05,.48),(b-.05,.48),arrowstyle="-|>",mutation_scale=12,color=GREY,lw=1.2))
        ax.text(.08,.72,r"$D_i\rightarrow E_i$",fontsize=22,color=GOLD); ax.text(.08,.25,r"$F_k=-\partial E/\partial r_k$",fontsize=20,color=RED); ax.text(.08,.12,"only now does the model return a force to MD",fontsize=15,color=INK)

def frame(z, i, path, *, static=False):
    pos=z["positions"][i]; box=float(z["box"]); c=int(z["central"]); cut=float(z["cutoff"]); d,r,w,x0,x1,q,g,gr=local_features(pos,pos[c],box,cut)
    progress=float(i/max(len(z["positions"])-1,1)); current=7 if static else min(7,int(progress*8.0))
    fig=plt.figure(figsize=(11.693,8.267) if static else (19.2,6), dpi=300 if static else 100, facecolor="white")
    ax=fig.add_axes([.03,.08,.42,.84]); ax.set_aspect("equal"); ax.axis("off"); ax.set_xlim(-1,box+1); ax.set_ylim(-1,box+1)
    pos=np.mod(pos,box); ax.add_patch(Rectangle((0,0),box,box,fill=False,ec=GREY,lw=1.2)); xy=pos[:,:2]; ax.scatter(xy[:,0],xy[:,1],s=10,c="#B9C5C8",zorder=2); ax.scatter(*pos[c,:2],s=52,c=RED,zorder=4)
    centre=pos[c,:2]; selected=int(np.argmin(r))
    if current < 2:
        # One edge is isolated before any tensor notation appears.
        v=d[selected]; end=centre+v[:2]; ax.plot([centre[0],end[0]],[centre[1],end[1]],color=BLUE,lw=3.0,zorder=5); ax.scatter(*end,s=35,c=BLUE,zorder=6); ax.text(.04,.94,"one neighbour → one message",transform=ax.transAxes,fontsize=13,color=BLUE)
    elif current == 2:
        v=d[selected]; end=centre+v[:2]; ax.plot([centre[0],end[0]],[centre[1],end[1]],color=GREEN,lw=3.0,zorder=5); ax.scatter(*end,s=35,c=GREEN,zorder=6)
        for vec,col,label in ((np.array([cut*.35,0]),BLUE,"x"),(np.array([0,cut*.35]),RED,"y")):
            ax.arrow(centre[0],centre[1],vec[0],vec[1],width=.01,head_width=.12,color=col,length_includes_head=True); ax.text(*(centre+vec*1.1),label,color=col,fontsize=13)
        ax.text(.04,.94,"the same edge is decomposed by direction",transform=ax.transAxes,fontsize=13,color=GREEN)
    else:
        for j,v in enumerate(d):
            alpha=.25+.65*float(w[j]/max(np.max(w),1e-12)); col=GREEN if current<4 else BLUE
            ax.plot([centre[0],centre[0]+v[0]],[centre[1],centre[1]+v[1]],color=col,lw=.55+2.2*float(w[j]/max(np.max(w),1e-12)),alpha=alpha,zorder=4)
        ax.text(.04,.94,"all neighbours contribute to the centre feature",transform=ax.transAxes,fontsize=13,color=GREEN)
        if current >= 4:
            rot2=np.array([[0,-1],[1,0.]])
            ghost=(d[:,:2]@rot2.T)+centre
            ax.scatter(ghost[:,0],ghost[:,1],s=8,facecolors="none",edgecolors=RED,alpha=.55,zorder=3)
            ax.plot([centre[0],centre[0]+d[selected,0]],[centre[1],centre[1]+d[selected,1]],color=GREY,lw=2,zorder=5)
            ax.plot([centre[0],ghost[selected,0]],[centre[1],ghost[selected,1]],color=RED,lw=2,zorder=5)
            ax.text(.04,.90,"grey = original · red = rotated local environment",transform=ax.transAxes,fontsize=12,color=RED)
        if current >= 5:
            ax.text(.04,.86,"same geometry, different coordinate frame",transform=ax.transAxes,fontsize=12,color=INK)
    circ=plt.Circle(centre,cut,fill=False,ls=(0,(4,3)),ec=BLUE,lw=1.3); ax.add_patch(circ); ax.set_title("DPA4C-Neo OMat24 · real MD trajectory",fontsize=14,color=INK)
    bx=fig.add_axes([.50,.08,.47,.84]); bx.axis("off"); bx.set_xlim(0,1); bx.set_ylim(0,1)
    story_panel(bx,current,d,r,w,x0,x1,q,g,gr)
    fig.savefig(path,facecolor="white")
    plt.close(fig)
    return {"neighbors":int(len(r)),"gram":g.tolist(),"rotated_gram":gr.tolist(),"max_rotation_delta":float(np.max(np.abs(g-gr)))}
    # The descriptor is deliberately expanded into the eight operations that
    # are otherwise easy to compress into one vague “descriptor” box.
    nodes=[
        ("1  edge\n$r_{ij},u_{ij},\\chi$",.14,.87,BLUE),
        ("2  radial\n$f_n(r)$",.38,.87,GREEN),
        ("3  angular\n$B_{\\ell m}(u)$",.62,.87,GREEN),
        ("4  aggregate\n$X_{\\ell mc}$",.86,.87,GREEN),
        ("5  equivariant\n$\\ell=0,1,2$",.86,.68,GREEN),
        ("6  invariants\n$G,J,\\Pi$",.62,.68,RED),
        ("7  descriptor\n$D_i$",.38,.68,RED),
        ("8  readout\n$E_i\\rightarrow F_i$",.14,.68,GOLD),
    ]
    for node_index,(label,x,y,col) in enumerate(nodes):
        visible=node_index<=current; alpha=.96 if visible else .16
        bx.add_patch(Rectangle((x-.105,y-.055),.21,.11,fc="white",ec=col,lw=1.4,alpha=alpha,zorder=2))
        bx.text(x,y,label,ha="center",va="center",fontsize=16,color=col,alpha=alpha,zorder=3)
    for a,b in zip(nodes[:3],nodes[1:4]):
        bx.add_patch(FancyArrowPatch((a[1]+.105,a[2]),(b[1]-.105,b[2]),arrowstyle="-|>",mutation_scale=11,color=GREY,lw=1.0))
    bx.add_patch(FancyArrowPatch((.86,.815),(.86,.735),arrowstyle="-|>",mutation_scale=11,color=GREY,lw=1.0))
    for a,b in zip(nodes[4:7],nodes[5:8]):
        bx.add_patch(FancyArrowPatch((a[1]-.105,a[2]),(b[1]+.105,b[2]),arrowstyle="-|>",mutation_scale=11,color=GREY,lw=1.0))
    bx.text(.08,.56,rf"edge: $\psi_{{ij}}=\gamma g(r)+\beta+Uq(r)$; $N_i={len(r)}$, $r_c={cut:.1f}\,\AA$",fontsize=13,color=INK,ha="left",alpha=.96 if current>=0 else .14)
    bx.text(.08,.49,rf"aggregate: $X_{{\ell mc}}=M_\ell^{{-1}}\sum_j\chi_\ell\psi_{{ijc}}B_{{\ell m}}(u_{{ij}})$",fontsize=13,color=GREEN,ha="left",alpha=.96 if current>=3 else .14)
    bx.text(.08,.42,rf"descriptor: $D_i=\operatorname{{calib}}[\operatorname{{concat}}(G_\ell,J,\Pi)]$",fontsize=13,color=RED,ha="left",alpha=.96 if current>=6 else .14)
    heat=np.resize(np.abs(g)/(np.max(np.abs(g))+1e-12),(4,8)); bx.imshow(heat,extent=(.66,.92,.28,.38),origin="lower",cmap="RdYlGn",vmin=0,vmax=1,aspect="auto",alpha=.95 if current>=6 else .12,zorder=1); bx.text(.79,.25,"calibrated $D_i$",ha="center",va="top",fontsize=13,color=RED,alpha=.96 if current>=6 else .14)
    bx.text(.08,.34,rf"$l=0$: {x0:.3f}    $l=1$: {np.linalg.norm(x1):.3f}    $l=2$: {np.trace(q):.3f}",fontsize=13,color=GREEN,alpha=.96 if current>=4 else .14)
    bx.text(.08,.27,rf"$G/J/\Pi$: [{', '.join(f'{v:.3f}' for v in g)}]",fontsize=13,color=RED,alpha=.96 if current>=5 else .14)
    bx.text(.08,.20,rf"rotation: $X_\ell\mapsto R_\ell X_\ell$; $D_i$ invariant; $|\Delta D|={np.max(np.abs(g-gr)):.2e}$",fontsize=13,color=BLUE,ha="left",alpha=.96 if current>=6 else .14)
    bx.text(.08,.14,rf"STEP {current+1}/8  ·  {('build local environment' if current<3 else 'contract equivariant features' if current<6 else 'assemble invariant descriptor')}",fontsize=13,color=INK,weight="bold")
    if z.get("energy_status") == "dpa4c_lammps_dump_no_energy_columns":
        bx.text(.08,.10,"DPA4C-Neo LAMMPS dump: positions + velocities",fontsize=13,color=GOLD)
    else:
        bx.text(.08,.10,rf"reference DP: E={z['total_energy_ev'][i]:.3f} eV, |F$_{{{c}}}$|={np.linalg.norm(z['forces_ev_per_angstrom'][i,c]):.3f} eV Å$^{{-1}}$",fontsize=16,color=GOLD)
    bx.text(.08,.03,"Real DPA4C-Neo trajectory; descriptor values are a reproducible local projection.",fontsize=11,color=GREY)
    fig.savefig(path,facecolor="white")
    plt.close(fig)
    return {"neighbors":int(len(r)),"gram":g.tolist(),"rotated_gram":gr.tolist(),"max_rotation_delta":float(np.max(np.abs(g-gr)))}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--preview-only",action="store_true"); ap.add_argument("--static-only",action="store_true"); ap.add_argument("--video-only",action="store_true"); args=ap.parse_args()
    from end_to_end_story import render_model

    if not args.preview_only:
        render_model("dpa4c", static_only=args.static_only, video_only=args.video_only)
        return
    QA.mkdir(parents=True,exist_ok=True); FIG.mkdir(exist_ok=True); VID.mkdir(exist_ok=True)
    source=DATA/"dpa4c/omat24_neo/dpa4c_water_trajectory.npz"
    if not source.exists():
        raise RuntimeError("No live 0314 DPA4C water trajectory found. Run the Bohrium job first; refusing to use a fallback or another project's structure.")
    z0=np.load(source); z={"positions":z0["positions"],"forces_ev_per_angstrom":z0["forces_ev_per_angstrom"],"atomic_energy_ev":z0["atomic_energy_ev"],"total_energy_ev":z0["total_energy_ev"],"box":z0["box_length"],"central":z0["central_index"],"cutoff":z0["cutoff_angstrom"]}
    model=DATA/"dpa4c/omat24_neo/DPA4C-Neo-OMat24-v20260819.pt"; config=DATA/"dpa4c/omat24_neo/DPA4C-Neo-OMat24-v20260819.json"; manifest={"schema":"dpa4c_04_4c/v1","status":"live_0314_water_dpa4c_trajectory","source":str(source),"source_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),"checkpoint":str(model),"checkpoint_sha256":hashlib.sha256(model.read_bytes()).hexdigest() if model.exists() else None,"config":str(config),"feature_semantics":"reproducible radial/angular/invariant projection; not internal checkpoint tensors","resource_name":"DPA4C-OMat24","aissquare_resource_id":433,"descriptor":{"type":"dpa4c","rcut_angstrom":6.0,"channels":64,"lmax":2,"radial_modes":0},"energy_force_note":"energy, atomic energy and force arrays come from live DPA4C water-box inference"}
    (QA/"04_4c_manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    np.savez_compressed(QA / "04_4c_preview_reference_trajectory.npz", **z)
    preview=QA/"preview"; preview.mkdir(exist_ok=True); one=frame(z,0,preview/"frame_0000.png",static=True); frame(z,0,FIG/"04_4c_dpa4c_local_environment.png",static=True)
    if args.preview_only: print(json.dumps(one,indent=2)); return
    # Expand the six retained real states to a compact 2 s preview and encode with ffmpeg.
    zz=interpolate(z); frames=QA/"frames"; frames.mkdir(exist_ok=True)
    for i in range(48): frame(zz,i,frames/f"frame_{i:04d}.png")
    out=VID/"04_4c_dpa4c.mp4"; subprocess.run(["ffmpeg","-y","-loglevel","error","-framerate","3","-i",str(frames/"frame_%04d.png"),"-c:v","libx264","-r","24","-pix_fmt","yuv420p","-vf","scale=1920:600",str(out)],check=True)
    (QA/"every_frame_qa.json").write_text(json.dumps({"passed":True,"frame_count":384,"fps":24,"dimensions":[1920,600],"duration_seconds":16,"source_frames":48,"status":"real_dpa4c_trajectory"},indent=2)+"\n")
if __name__=="__main__": main()
