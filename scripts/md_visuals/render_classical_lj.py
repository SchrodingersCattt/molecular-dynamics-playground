"""Classical water-dimer story: measure geometry, then evaluate the LJ force."""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle, Arc
from PIL import Image
from common import DARK_GRAY, INK, LINE_GRAY, NAVY, LayoutRegistry, new_static_figure, render_video, save_static
from mattervis_story import camera_for_source, draw_world_segment, make_vector_group, render_structure
from responsive_story import EMERALD, LAKE_BLUE, PALE_OLIVE, panel_box, simple_audit, stage_rail, story_axes

ROOT=Path(__file__).resolve().parents[2]/"product"; STEM="02_classical_lj"; QA_DIR=ROOT/"qa"/STEM
SOURCE=ROOT/"data"/"classical_lj_trajectory.extxyz"; ASSET_DIR=QA_DIR/"source"/"mattervis_v4"
ARROW_STYLE={"shaft_radius":.018,"head_length":.065,"head_radius":.042,"sides":18}
def load_data():
    with np.load(ROOT/"data"/"classical_lj.npz",allow_pickle=True) as z:return {k:z[k] for k in z.files}
def _render(out,camera,frame,vectors=None):
    render_structure(SOURCE,out,camera=camera,frame=frame,view="cluster",width=1700,height=1180,atom_scale=1.06,bond_radius=.135,vector_overlays=vectors or [])
def prepare_mattervis(d):
    p=np.asarray(d["trajectory_atomic_positions"]); q=np.asarray(d["trajectory_oxygen_positions"]); f=np.asarray(d["trajectory_forces"]); v=np.asarray(d["trajectory_velocities"]); cam=camera_for_source(SOURCE,target=p.mean(axis=(0,1)),ortho_scale=4.0,frame=0,direction=(.12,-.82,.56),up=(0,0,1)); ASSET_DIR.mkdir(parents=True,exist_ok=True); plain=[];force=[];vel=[]
    for i in range(len(p)):
        a=[ASSET_DIR/f"{x}_{i:02d}.png" for x in ("plain","force","velocity")]
        if not a[0].exists():_render(a[0],cam,i)
        if not a[1].exists():_render(a[1],cam,i,make_vector_group("lj-F",q[i],f[i],scale=4,color=PALE_OLIVE,style=ARROW_STYLE))
        if not a[2].exists():_render(a[2],cam,i,make_vector_group("lj-v",q[i],v[i],scale=55,color=EMERALD,style=ARROW_STYLE))
        plain.append(a[0]);force.append(a[1]);vel.append(a[2])
    boxes=[]
    for path in plain:
        with Image.open(path) as im:
            m=np.asarray(im.convert("RGBA"))[:,:,3]>8;y,x=np.where(m);boxes.append((x.min(),y.min(),x.max()+1,y.max()+1))
    return {"plain":plain,"force":force,"velocity":vel,"camera":cam,"crop":(min(b[0] for b in boxes),min(b[1] for b in boxes),max(b[2] for b in boxes),max(b[3] for b in boxes))}
def fixed_place(ax,path,rect,crop,alpha=1,zorder=10):
    with Image.open(path) as im:image=np.asarray(im.convert("RGBA"))[crop[1]:crop[3],crop[0]:crop[2]]
    x0,y0,x1,y1=rect; ax.imshow(image,extent=(x0,x1,y0,y1),origin="upper",interpolation="lanczos",alpha=alpha,zorder=zorder,aspect="auto")
def projection_rect(rect,crop):
    x0,y0,x1,y1=rect; cx0,cy0,cx1,cy1=crop; w=x1-x0; h=y1-y0
    return (x0-cx0/(cx1-cx0)*w,y0-cy0/(cy1-cy0)*h,x0+(1700-cx0)/(cx1-cx0)*w,y0+(1180-cy0)/(cy1-cy0)*h)
def geometry_overlay(ax,reg,d,scenes,stage,video=True):
    rect=(.03,.13,.97,.90); fixed_place(ax,scenes["plain"][0],rect,scenes["crop"],.20)
    q=np.asarray(d["trajectory_atomic_positions"])[0]; cam=scenes["camera"]; fitted=projection_rect(rect,scenes["crop"])
    if stage==0:
        xy=draw_world_segment(ax,q[0],q[1],camera=cam,rect=fitted,color=LAKE_BLUE,linewidth=4 if video else 2.5,image_aspect=1700/1180)
        reg.text(ax,float(xy[:,0].mean()),float(xy[:,1].mean()+.05),r"$r_{OH}=0.957$ Å",ha="center",va="bottom",fontsize=11 if video else 10,color=LAKE_BLUE,weight="bold")
    if stage==1:
        xy1=draw_world_segment(ax,q[0],q[1],camera=cam,rect=fitted,color=PALE_OLIVE,linewidth=4 if video else 2.5,image_aspect=1700/1180)
        xy2=draw_world_segment(ax,q[0],q[2],camera=cam,rect=fitted,color=PALE_OLIVE,linewidth=4 if video else 2.5,image_aspect=1700/1180)
        vertex=xy1[0]; reg.text(ax,float(vertex[0]+.08),float(vertex[1]),"angle HOH = 104.5 deg",ha="left",va="center",fontsize=11 if video else 10,color=PALE_OLIVE,weight="bold")
    if stage==2:
        xy=draw_world_segment(ax,q[0],q[3],camera=cam,rect=fitted,color=NAVY,linewidth=4 if video else 2.5,image_aspect=1700/1180)
        reg.text(ax,float(xy[:,0].mean()),float(xy[:,1].mean()+.05),r"$r_{OO}=2.900$ Å",ha="center",va="bottom",fontsize=11 if video else 10,color=NAVY,weight="bold")
def info(ax,reg,d,mode,video=True):
    panel_box(ax,reg,"GEOMETRY → LJ FORCE",video=video)
    rows=[("geometry","rOH = 0.957 A",LAKE_BLUE),("geometry","angle HOH = 104.5 deg",PALE_OLIVE),("LJ input","rOO = 2.900 A",NAVY),("potential","U_LJ(rOO)",EMERALD),("force","-dU/dr -> FOO",PALE_OLIVE)]
    for i,(label,value,color) in enumerate(rows):
        y=.84-i*.16; active=(mode=="geometry" and i<=2) or (mode=="distance" and i==2) or (mode=="energy" and i==3) or (mode in {"force","accel"} and i==4)
        ax.add_patch(FancyBboxPatch((.08,y-.055),.84,.11,boxstyle="round,pad=.008,rounding_size=.015",fc="#EAF2EE" if active else "#F7F8F6",ec=color if active else LINE_GRAY,lw=2 if video else 1.3)); reg.text(ax,.16,y+.032,label,ha="left",va="center",fontsize=10 if video else 9,color=color,weight="bold"); reg.text(ax,.16,y-.032,value,ha="left",va="center",fontsize=10 if video else 9,color=INK)
    reg.text(ax,.5,.10,"O–O LJ term only · electrostatics omitted",ha="center",va="center",fontsize=9 if video else 8,color=DARK_GRAY)
def state_at(t):
    if t<2:return "structure",0
    if t<5:return "geometry",(t-2)/3
    if t<7:return "distance",1
    if t<9:return "energy",(t-7)/2
    if t<11:return "force",(t-9)/2
    if t<13:return "accel",(t-11)/2
    if t<16:return "velocity",(t-13)/3
    if t<17:return "position",1
    if t<18:return "commit",1
    if t<23:return "fast",(t-18)/5
    return "commit",1
def compose(fig,t,reg,d,s,video=True):
    mode,prog=state_at(t);rail,main,right=story_axes(fig,video=video);active={"force":1,"accel":1,"velocity":2,"position":0}.get(mode);stage_rail(rail,reg,active=active,video=video,return_phase=mode=="commit");title={"structure":"initial water-dimer structure","geometry":"measure molecular geometry","distance":"lock O···O distance","energy":"evaluate LJ potential","force":"LJ force: opposite pair","accel":"force → acceleration","velocity":"update velocity","position":"update position","commit":"complete state","fast":"rapid VV steps"}[mode];panel_box(main,reg,f"CLASSICAL MD · TIP3P WATER DIMER · {title}",video=video);rect=(.03,.13,.97,.90)
    if mode in {"structure","geometry","distance"}:
        gstage = 0 if mode == "structure" else (min(2, int(prog * 3.0)) if mode == "geometry" else 2)
        geometry_overlay(main,reg,d,s,gstage,video=video)
    elif mode in {"energy","force","accel"}: fixed_place(main,s["plain"][0],rect,s["crop"],.16);fixed_place(main,s["force"][0],rect,s["crop"] if mode!="energy" else s["crop"],1 if mode!="energy" else .12);reg.text(main,.035,.075,r"$U_{LJ}(r)=4\epsilon[(\sigma/r)^{12}-(\sigma/r)^6]$" if mode=="energy" else r"$\mathbf{F}_{OO}=-\nabla U_{LJ}$",ha="left",va="bottom",fontsize=11 if video else 10,color=PALE_OLIVE if mode!="energy" else EMERALD,weight="bold")
    elif mode=="velocity":fixed_place(main,s["plain"][0],rect,s["crop"],.16);fixed_place(main,s["velocity"][0],rect,s["crop"]);reg.text(main,.035,.075,r"$v_{n+1}=v_n+\frac{\Delta t}{2m}(F_n+F_{n+1})$",ha="left",va="bottom",fontsize=11 if video else 10,color=EMERALD,weight="bold")
    elif mode=="position":fixed_place(main,s["plain"][0],rect,s["crop"],.20);fixed_place(main,s["plain"][1],rect,s["crop"]);reg.text(main,.035,.075,"velocity → new position",ha="left",va="bottom",fontsize=11 if video else 10,color=LAKE_BLUE,weight="bold")
    elif mode=="fast":idx=min(6,1+int(prog*5));fixed_place(main,s["plain"][idx],rect,s["crop"]);reg.text(main,.035,.075,f"real trajectory · step {idx}",ha="left",va="bottom",fontsize=11 if video else 10,color=DARK_GRAY)
    else:fixed_place(main,s["plain"][1],rect,s["crop"])
    info(right,reg,d,mode,video=video);return [{"id":"distance","color":NAVY,"min_pixels":60}] if mode in {"geometry","distance"} else ([{"id":"force","color":PALE_OLIVE,"min_pixels":80}] if mode in {"force","accel"} else [])
def render_static(d,s):
    fig=new_static_figure();reg=LayoutRegistry(min_font_pt=10,max_font_pt=16,edge_pad_px=18);compose(fig,4,reg,d,s,video=False);errors=reg.validate(fig)
    if errors:raise RuntimeError("static responsive layout failed:\n"+"\n".join(errors))
    save_static(fig,STEM)
def render_animation(d,s):
    render_video(stem=STEM,duration_seconds=24,draw_frame=lambda f,t,i,r:compose(f,t,r,d,s),audit_config=simple_audit(("rail","structure","lj_info")),qa_directory=QA_DIR/"_qa",representative_times=[1,2.5,4,6,8,10,12,14.5,16.5,19,23.5])
def main():
    p=argparse.ArgumentParser();p.add_argument("--static-only",action="store_true");p.add_argument("--video-only",action="store_true");a=p.parse_args();d=load_data();s=prepare_mattervis(d)
    if not a.video_only:render_static(d,s)
    if not a.static_only:render_animation(d,s)
if __name__=="__main__":main()
