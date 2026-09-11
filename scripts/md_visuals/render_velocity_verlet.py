"""Velocity-Verlet teaching story: initialise r/v, then close the VV loop."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image
from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static
from mattervis_story import camera_for_source, make_vector_group, render_structure, project_world, draw_world_segment
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, panel_box, simple_audit, stage_rail, story_axes

ROOT=Path(__file__).resolve().parents[2]/"product"; STEM="01_velocity_verlet"; QA_DIR=ROOT/"qa"/STEM
SOURCE=ROOT/"data"/"vv_h2o_trajectory.extxyz"; ASSET_DIR=QA_DIR/"source"/"mattervis_v11"
ARROW_STYLE={"shaft_radius":.010,"head_length":.040,"head_radius":.022,"sides":18}

def load_data():
    with np.load(ROOT/"data"/"vv_h2o_step.npz",allow_pickle=True) as z:return {k:z[k] for k in z.files}
def _render(source,out,camera,frame,vectors=None):
    render_structure(source,out,camera=camera,frame=frame,view="cluster",width=1700,height=1180,atom_scale=1.10,bond_radius=.14,vector_overlays=vectors or [])
def prepare_mattervis(data):
    ASSET_DIR.mkdir(parents=True,exist_ok=True); p=np.asarray(data["trajectory_positions"]); v=np.asarray(data["trajectory_velocities"]); a=np.asarray(data["trajectory_accelerations"]); f=np.asarray(data["trajectory_forces"]); cam=camera_for_source(SOURCE,target=p.mean(axis=(0,1)),ortho_scale=1.30,frame=0); plain=[];vel=[];acc=[];force=[]
    for i in range(len(p)):
        paths=[ASSET_DIR/f"{x}_{i:02d}.png" for x in ("plain","velocity","acceleration","force")]
        if not paths[0].exists():_render(SOURCE,paths[0],cam,i)
        if not paths[1].exists():_render(SOURCE,paths[1],cam,i,make_vector_group("vv-v",p[i],v[i],scale=8,color=EMERALD,style=ARROW_STYLE))
        if not paths[2].exists():_render(SOURCE,paths[2],cam,i,make_vector_group("vv-a",p[i],a[i],scale=12,color=PALE_OLIVE,style=ARROW_STYLE))
        if not paths[3].exists():
            _render(SOURCE,paths[3],cam,i,make_vector_group("vv-f",p[i],f[i],scale=1.6,color="#A44738",style=ARROW_STYLE))
        plain.append(paths[0]);vel.append(paths[1]);acc.append(paths[2]);force.append(paths[3])
    force_steps=[]
    for k in range(3):
        out=ASSET_DIR/f"force_partial_{k}.png"
        if not out.exists():
            _render(SOURCE,out,cam,0,make_vector_group("vv-f-partial",p[0,:k+1],f[0,:k+1],scale=1.6,color="#A44738",style=ARROW_STYLE))
        force_steps.append(out)
    assignments=[]
    for k in range(3):
        out=ASSET_DIR/f"velocity_assign_{k}.png"
        if not out.exists():
            mask=v[0,:k+1]
            _render(SOURCE,out,cam,0,make_vector_group("vv-init-v",p[0,:k+1],mask,scale=8,color=EMERALD,style=ARROW_STYLE))
        assignments.append(out)
    boxes=[]
    # Use only the fixed molecular geometry to determine the crop.  Vector
    # heads are an explanatory overlay and must not resize the molecule.
    for path in plain:
        with Image.open(path) as im:
            m=np.asarray(im.convert("RGBA"))[:,:,3]>8; y,x=np.where(m); boxes.append((x.min(),y.min(),x.max()+1,y.max()+1))
    x0,y0,x1,y1=(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes)); px=int(.16*(x1-x0)); py=int(.16*(y1-y0)); crop=(max(0,x0-px),max(0,y0-py),min(1700,x1+px),min(1180,y1+py))
    # A true vector-addition scene: old velocity arrows remain rooted at the
    # atoms; the delta-v arrows begin at their displayed tips.
    add_path=ASSET_DIR/"velocity_add.png"
    if not add_path.exists():
        dt=float(data["dt_fs"]); delta=.5*(a[0]+a[1])*dt; scale=8.0
        origins=np.vstack((p[0],p[0]+scale*v[0]))
        vectors=np.vstack((v[0],delta))
        _render(SOURCE,add_path,cam,0,[make_vector_group("vv-v-old",p[0],v[0],scale=scale,color=EMERALD,style=ARROW_STYLE)[0],make_vector_group("vv-dv",p[0]+scale*v[0],delta,scale=scale,color=PALE_OLIVE,style=ARROW_STYLE)[0]])
    disp_path=ASSET_DIR/"displacement.png"
    if not disp_path.exists():
        _render(SOURCE,disp_path,cam,0,make_vector_group("vv-dr",p[0],p[1]-p[0],scale=24,color=LAKE_BLUE,style=ARROW_STYLE))
    return {"plain":plain,"velocity":vel,"acceleration":acc,"force":force,"force_steps":force_steps,"assignments":assignments,"velocity_add":add_path,"displacement":disp_path,"camera":cam,"crop":crop}
def fixed_place(ax,path,rect,crop,alpha=1,zorder=10):
    with Image.open(path) as im: image=np.asarray(im.convert("RGBA"))[crop[1]:crop[3],crop[0]:crop[2]]
    x0,y0,x1,y1=rect; ax.imshow(image,extent=(x0,x1,y0,y1),origin="upper",interpolation="lanczos",alpha=alpha,zorder=zorder,aspect="auto")
def projection_rect(rect,crop):
    x0,y0,x1,y1=rect; cx0,cy0,cx1,cy1=crop; w=x1-x0; h=y1-y0
    return (x0-cx0/(cx1-cx0)*w,y0-cy0/(cy1-cy0)*h,x0+(1700-cx0)/(cx1-cx0)*w,y0+(1180-cy0)/(cy1-cy0)*h)
def coordinate_overlay(ax,reg,data,scenes,video=True):
    q=np.asarray(data["trajectory_positions"])[0]; rect=(.03,.13,.97,.90); proj=projection_rect(rect,scenes["crop"]); xy=project_world(q,camera=scenes["camera"],rect=proj,image_aspect=1700/1180)
    offsets=[(-.18,-.18),(.10,.12),(.10,-.14)]
    for i,(name,point,(dx,dy)) in enumerate(zip(("O","H1","H2"),xy,offsets)):
        target=np.array(point)+np.array((dx,dy)); ax.plot([point[0],target[0]],[point[1],target[1]],color=NAVY,lw=1.5 if video else 1,zorder=25)
        xyz=q[i]; label=f"{name}\nx={xyz[0]:+.3f} A\ny={xyz[1]:+.3f} A\nz={xyz[2]:+.3f} A"
        reg.text(ax,float(target[0]),float(target[1]),label,ha="center",va="center",fontsize=9 if video else 8,color=NAVY,weight="bold",zorder=30)
def velocity_chart(ax,reg,data,progress,video=True):
    panel_box(ax,reg,"INITIAL VELOCITY",video=video)
    reg.text(ax,.5,.88,"T = 1200 K  ·  Maxwell–Boltzmann draw",ha="center",va="center",fontsize=10 if video else 9,color=NAVY,weight="bold")
    speeds=np.linalg.norm(np.asarray(data["trajectory_velocities"])[0],axis=1); masses=np.asarray(data["masses"],dtype=float)
    rng=np.random.default_rng(20260827); samples=np.concatenate([np.linalg.norm(rng.normal(0,np.sqrt(8.617333e-5*1200*.0096485/m),size=(32,3)),axis=1) for m in masses])
    hi=max(float(np.quantile(samples,.98)),float(speeds.max())*1.1,1e-6); bins=np.linspace(0,hi,19); counts,_=np.histogram(samples,bins=bins); max_count=max(int(counts.max()),1); width=.74/18
    ax.add_patch(Rectangle((.13,.28),.74,.48,fc="#F5F7F6",ec=LINE_GRAY,lw=1.2))
    for j,count in enumerate(counts):
        x=.13+j*width; h=.40*count/max_count
        ax.add_patch(Rectangle((x+.006,.31),width-.012,h,fc="#B6C2C2",ec="white",lw=.4))
    colors=("#A44738",PALE_OLIVE,EMERALD); names=("O","H1","H2"); assigned=min(3,int(np.floor(progress*3+1e-9)))
    for i,(name,speed,color) in enumerate(zip(names,speeds,colors)):
        if i<assigned or (i==assigned and assigned<3 and progress*3-assigned>0):
            j=min(17,int(np.searchsorted(bins,speed,side="right")-1)); x=.13+j*width
            ax.add_patch(Rectangle((x+.006,.31),width-.012,.40*max(.18,counts[j])/max_count,fc=color,ec=color,lw=1.5))
        reg.text(ax,.18+i*.31,.20,name,ha="center",va="center",fontsize=10 if video else 9,color=color,weight="bold")
    reg.text(ax,.5,.10,"highlighted bars → vectors → O / H1 / H2",ha="center",va="center",fontsize=9 if video else 8,color=DARK_GRAY)
def info_panel(ax,reg,data,mode,progress,video=True):
    if mode=="init":velocity_chart(ax,reg,data,progress,video=video);return
    if mode in {"force","accel"}:
        panel_box(ax,reg,"FORCE → ACCELERATION",video=video)
        forces=np.asarray(data["trajectory_forces"])[0]; acc=np.asarray(data["trajectory_accelerations"])[0]
        for i,name in enumerate(("O","H1","H2")):
            y=.72-.19*i; color="#A44738" if mode=="force" else PALE_OLIVE
            active=i <= min(2,int(progress*3.0)) if mode=="force" else True
            ax.add_patch(FancyBboxPatch((.08,y-.06),.84,.12,boxstyle="round,pad=.008,rounding_size=.015",fc="#EAF2EE" if active else "#F7F8F6",ec=color if active else LINE_GRAY,lw=2 if video else 1.3))
            reg.text(ax,.16,y+.025,name,ha="left",va="center",fontsize=10 if video else 9,color=color,weight="bold")
            reg.text(ax,.32,y+.025,f"|F|={np.linalg.norm(forces[i]):.2f}",ha="left",va="center",fontsize=9 if video else 8,color=INK)
            reg.text(ax,.32,y-.030,f"|a|={np.linalg.norm(acc[i]):.4f}",ha="left",va="center",fontsize=9 if video else 8,color=INK)
        reg.text(ax,.5,.10,"a_i = F_i / m_i",ha="center",va="center",fontsize=10 if video else 9,color=NAVY,weight="bold"); return
    panel_box(ax,reg,"STATE",video=video); cards=[("known",r"$r_n,\;v_n$",LAKE_BLUE),("new",r"$r^*,\;v^*,\;a^*$",EMERALD)]
    for i,(label,value,color) in enumerate(cards):
        y=.72-i*.27; selected=(mode in {"position","lock","force"} and i==0) or (mode in {"velocity","commit"} and i==1); ax.add_patch(FancyBboxPatch((.08,y-.085),.84,.17,boxstyle="round,pad=.012,rounding_size=.02",fc="#EAF2EE" if selected else "#F7F8F6",ec=color if selected else LINE_GRAY,lw=2.2 if video else 1.4)); reg.text(ax,.18,y+.03,label,ha="left",va="center",fontsize=11 if video else 10,color=color,weight="bold"); reg.text(ax,.18,y-.03,value,ha="left",va="center",fontsize=10 if video else 9,color=INK)
    reg.text(ax,.5,.20,r"$\Delta t=0.5\,\mathrm{fs}$",ha="center",va="center",fontsize=11 if video else 10,color=NAVY,weight="bold"); reg.text(ax,.5,.12,"one real H2O trajectory",ha="center",va="center",fontsize=10 if video else 9,color=DARK_GRAY)
    if mode=="commit":reg.text(ax,.5,.055,r"$n\rightarrow n+1$",ha="center",va="center",fontsize=11 if video else 10,color=NAVY,weight="bold")
def state_at(t):
    if t<1:return "position",t
    if t<2:return "init_hist",0
    if t<3:return "init_assign",t-2
    if t<4:return "lock",1
    if t<7:return "force",(t-4)/3
    if t<9:return "velocity",(t-7)/2
    if t<11:return "position",(t-9)/2
    if t<12:return "commit",1
    if t<19:return "fast",(t-12)/7
    return "commit",1
def compose(fig,t,reg,data,scenes,video=True):
    mode,progress=state_at(t); rail,main,info=story_axes(fig,video=video); active={"position":0,"force":1,"velocity":2}.get(mode); stage_rail(rail,reg,active=active,video=video,equation=None,return_phase=mode=="commit"); title={"position":"initial position / update position","init_hist":"given velocity distribution","init_assign":"assign velocity to each atom","lock":"initial state ready","force":"calculate force → acceleration","velocity":"update velocity","commit":"complete state","fast":"rapid VV steps"}[mode]; panel_box(main,reg,f"VELOCITY–VERLET · H2O · {title}",video=video); crop=scenes["crop"]; rect=(.03,.13,.97,.90)
    if mode=="init_hist":fixed_place(main,scenes["plain"][0],rect,crop,.22);reg.text(main,.035,.075,"given distribution · ready to assign",ha="left",va="bottom",fontsize=11 if video else 10,color=DARK_GRAY)
    elif mode=="init_assign":fixed_place(main,scenes["plain"][0],rect,crop,.22);fixed_place(main,scenes["assignments"][min(2,int(progress*3))],rect,crop);reg.text(main,.035,.075,"assign O → H1 → H2",ha="left",va="bottom",fontsize=11 if video else 10,color=EMERALD,weight="bold")
    elif mode=="init_hist":fixed_place(main,scenes["plain"][0],rect,crop,.22);reg.text(main,.035,.075,"given distribution · ready to assign",ha="left",va="bottom",fontsize=11 if video else 10,color=DARK_GRAY)
    elif mode=="force":
        fixed_place(main,scenes["plain"][0],rect,crop,.18)
        if progress < .70: fixed_place(main,scenes["force_steps"][min(2,int(progress/.70*3))],rect,crop)
        else: fixed_place(main,scenes["acceleration"][0],rect,crop)
        reg.text(main,.035,.075,"F_i(r_n) → a_i = F_i / m_i",ha="left",va="bottom",fontsize=11 if video else 10,color=PALE_OLIVE,weight="bold")
    elif mode=="velocity":
        fixed_place(main,scenes["plain"][0],rect,crop,.16)
        fixed_place(main,scenes["velocity"][0] if progress < .30 else scenes["velocity_add"],rect,crop)
        reg.text(main,.035,.075,r"$v_{n+1}=v_n+\frac{1}{2}(a_n+a_{n+1})\Delta t$",ha="left",va="bottom",fontsize=11 if video else 10,color=EMERALD,weight="bold")
    elif mode=="position":
        if t < 1:
            fixed_place(main,scenes["plain"][0],rect,crop)
            coordinate_overlay(main,reg,data,scenes,video=video)
        elif t < 10: fixed_place(main,scenes["plain"][0],rect,crop,.18); fixed_place(main,scenes["displacement"],rect,crop)
        else: fixed_place(main,scenes["plain"][0],rect,crop,.25); fixed_place(main,scenes["plain"][1],rect,crop)
        reg.text(main,.035,.075,"position = molecular structure" if t < 1 else r"$\Delta r_i$ → new structure $r_{n+1}$",ha="left",va="bottom",fontsize=11 if video else 10,color=LAKE_BLUE,weight="bold")
    elif mode=="fast":idx=min(6,1+int(progress*5));fixed_place(main,scenes["plain"][idx],rect,crop);reg.text(main,.035,.075,f"real trajectory · step {idx}",ha="left",va="bottom",fontsize=11 if video else 10,color=DARK_GRAY)
    else:fixed_place(main,scenes["plain"][1],rect,crop)
    chart_mode="init" if mode in {"init_hist","init_assign"} else ("accel" if mode=="force" and progress>=.70 else mode); chart_progress=0 if mode=="init_hist" else progress
    info_panel(info,reg,data,chart_mode,chart_progress,video=video)
    if mode == "force" and progress < .70: return [{"id":"force","color":"#A44738","min_pixels":80}]
    if mode == "force": return [{"id":"acceleration","color":PALE_OLIVE,"min_pixels":80}]
    if mode == "velocity": return [{"id":"velocity","color":EMERALD,"min_pixels":80}]
    return []
def render_static(data,scenes):
    fig=new_static_figure();reg=LayoutRegistry(min_font_pt=10,max_font_pt=16,edge_pad_px=18);compose(fig,0,reg,data,scenes,video=False);errors=reg.validate(fig)
    if errors:raise RuntimeError("static responsive layout failed:\n"+"\n".join(errors))
    save_static(fig,STEM)
def render_animation(data,scenes):
    render_video(stem=STEM,duration_seconds=20,draw_frame=lambda f,t,i,r:compose(f,t,r,data,scenes),audit_config=simple_audit(("rail","structure","state")),qa_directory=QA_DIR/"_qa",representative_times=[.5,2,4.5,6.5,8.5,10.5,12.5,15,19.5])
def main():
    p=argparse.ArgumentParser();p.add_argument("--static-only",action="store_true");p.add_argument("--video-only",action="store_true");a=p.parse_args();data=load_data();scenes=prepare_mattervis(data)
    if not a.video_only:render_static(data,scenes)
    if not a.static_only:render_animation(data,scenes)
if __name__=="__main__":main()
