"""Synthesize a SYMMETRIC pincha (feathered-peacock / forearm-stand) reference.

The mocap pincha takes are all ONE-ARMED: the performer plants the LEFT forearm
and floats the RIGHT forearm 3-4 cm the whole hold (measured across -a/-b/-c/-d;
lr_symmetry ~0.27-0.30). That precarious asymmetric base is what makes the node
hard to initialize on the soft asset -- NOT ground penetration (the trims seat
clean to <=0.5 mm). Fix per user direction: mirror the PLANTED LEFT side onto the
right so both forearms plant symmetrically.

Method (per frame of the forearm-stand hold window of the source clip):
  * paired joints (L/R hip,knee,ankle,toe,thorax,shoulder,elbow,wrist,hand):
    RIGHT := mirror(LEFT); LEFT kept. Both sides now driven by the good left side.
    (Averaging LEFT with mirror(RIGHT) would leave BOTH forearms floating ~2 cm --
    we want the planted height, so we COPY the left, not average.)
  * midline joints (Torso/Spine/Chest/Neck/Head) + root_rot: slerp(orig, mirror, 0.5)
    -> removes the ~16 deg pelvis/spine twist.
Sagittal mirror (left axis = world y): exp-map (vx,vy,vz) -> (-vx,vy,-vz);
root_pos y -> -y. Rest skeleton is bilaterally symmetric so mirrored LOCAL
rotations produce a mirrored WORLD pose.

Frame layout (75): [root_pos(3), root_rot expmap(3), 23 joints x 3 expmap(69)];
joint col for body i>=1 is 6+3*(i-1).

Run:
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/make_pincha_synth_clip.py \
     --src data/motions/smpl/220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a \
     --out data/motions/smpl/pincha_synth
"""
import argparse, os, sys
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_REPO)
sys.path.insert(0, os.path.join(_REPO, "tools"))
sys.path.insert(0, os.path.join(_REPO, "mimickit"))
import numpy as np
import torch
import anim.kin_char_model as kin_char_model
import anim.motion as motion
import anim.char_geoms as char_geoms
import util.torch_util as torch_util
import annotate_clips as A

# body-index L/R pairs and midline joints (from smpl_boxhands ordering)
PAIRS = [(1, 5), (2, 6), (3, 7), (4, 8), (14, 19), (15, 20), (16, 21), (17, 22), (18, 23)]
MIDLINE = [9, 10, 11, 12, 13]     # Torso, Spine, Chest, Neck, Head
def col(bi): return 6 + 3 * (bi - 1)

def mirror_exp(v):                 # sagittal (y) mirror of an exp-map: (-x, y, -z)
    return torch.stack([-v[..., 0], v[..., 1], -v[..., 2]], dim=-1)

def slerp(q0, q1, t):
    q0 = q0 / q0.norm(); q1 = q1 / q1.norm()
    d = (q0 * q1).sum()
    if d < 0: q1 = -q1; d = -d
    if d > 0.9995:
        q = q0 + t * (q1 - q0); return q / q.norm()
    th0 = torch.acos(d.clamp(-1, 1)); th = th0 * t
    q2 = q1 - q0 * d; q2 = q2 / q2.norm()
    return q0 * torch.cos(th) + q2 * torch.sin(th)

def symmetrize_frame(f):
    s = f.clone()
    # root position: mirror-neutralize the y (keep x,z) -> centered
    s[1] = 0.0
    # root rot: untwist via slerp with its sagittal mirror
    q = torch_util.exp_map_to_quat(f[3:6].unsqueeze(0))[0]
    qm = torch_util.exp_map_to_quat(mirror_exp(f[3:6]).unsqueeze(0))[0]
    s[3:6] = torch_util.quat_to_exp_map(slerp(q, qm, 0.5).unsqueeze(0))[0]
    # midline joints: untwist via slerp with mirror
    for bi in MIDLINE:
        c = col(bi)
        q = torch_util.exp_map_to_quat(f[c:c+3].unsqueeze(0))[0]
        qm = torch_util.exp_map_to_quat(mirror_exp(f[c:c+3]).unsqueeze(0))[0]
        s[c:c+3] = torch_util.quat_to_exp_map(slerp(q, qm, 0.5).unsqueeze(0))[0]
    # paired joints: RIGHT := mirror(LEFT); LEFT kept
    for (lb, rb) in PAIRS:
        lc, rc = col(lb), col(rb)
        s[lc:lc+3] = f[lc:lc+3]
        s[rc:rc+3] = mirror_exp(f[lc:lc+3])
    return s

def fk_all(kin, frames):
    rp = frames[:, 0:3]
    rr = torch_util.exp_map_to_quat(frames[:, 3:6])
    jr = torch_util.exp_map_to_quat(frames[:, 6:].reshape(frames.shape[0], -1, 3))
    return kin.forward_kinematics(rp, rr, jr)

def longest_run(mask):
    best=(0,0); i=0; n=len(mask)
    while i<n:
        if not mask[i]: i+=1; continue
        j=i
        while j+1<n and mask[j+1]: j+=1
        if j-i>best[1]-best[0]: best=(i,j)
        i=j+1
    return best

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--src", default="data/motions/smpl/220923_Feathered_Peacock_Pose_or_Pincha_Mayurasana_-a")
    p.add_argument("--char_file", default="data/assets/smpl/smpl_boxhands.xml")
    p.add_argument("--out", default="data/motions/smpl/pincha_synth")
    p.add_argument("--inv_thresh", type=float, default=-0.88)   # up_z below this = forearm stand
    p.add_argument("--max_s", type=float, default=16.0)         # cap window length
    args=p.parse_args()

    kin=kin_char_model.KinCharModel("cpu"); kin.load_char_file(args.char_file)
    BN=kin.get_body_names(); GE=char_geoms.load_char_geoms(args.char_file, BN, "cpu")
    idx={n:BN.index(n) for n in BN}

    raw=motion.load_motion(args.src)
    frames=torch.tensor(np.asarray(raw.frames, dtype=np.float32)); fps=float(raw.fps)
    bp,br=fk_all(kin, frames)
    up=torch.zeros(frames.shape[0],3); up[:,2]=1.0
    up_z=torch_util.quat_rotate(torch_util.exp_map_to_quat(frames[:,3:6]), up)[:,2]
    i0,i1=longest_run((up_z<args.inv_thresh).tolist())
    if (i1-i0)/fps>args.max_s:               # center-crop to max_s
        c=(i0+i1)//2; half=int(args.max_s*fps/2); i0,i1=c-half,c+half
    win=frames[i0:i1+1]
    print("source {}  fps {:.0f}".format(os.path.basename(args.src), fps))
    print("forearm-stand window: frames {}..{} ({:.2f}s), up_z mean {:+.3f}".format(
        i0,i1,(i1-i0)/fps,float(up_z[i0:i1+1].mean())))

    sym=torch.stack([symmetrize_frame(win[t]) for t in range(win.shape[0])],0)

    up1=torch.zeros(1,3); up1[0,2]=1.0
    def seated_report(fr, tag):
        b,brr=fk_all(kin, fr)
        minz,_=A.per_body_witness(GE, b, brr)
        off=float((-minz.amin()).clamp(min=0)); z=minz+off
        upz=torch_util.quat_rotate(torch_util.exp_map_to_quat(fr[:,3:6]), up1.expand(fr.shape[0],3))[:,2]
        def c(n): return float(z[:,idx[n]].mean())
        feet=float(b[:,[idx["L_Toe"],idx["R_Toe"]],2].mean())
        print("  [{:9s}] up_z {:+.3f} feet_up {:.2f}m off {:.3f} | "
              "L_Elb {:+.3f} R_Elb {:+.3f} (Δ{:.3f}) | L_Hand {:+.3f} R_Hand {:+.3f} (Δ{:.3f})".format(
            tag, float(upz.mean()), feet, off,
            c("L_Elbow"),c("R_Elbow"),abs(c("L_Elbow")-c("R_Elbow")),
            c("L_Hand"),c("R_Hand"),abs(c("L_Hand")-c("R_Hand"))))
    print("-"*92)
    seated_report(win, "original")
    seated_report(sym, "SYMMETRIC")

    out=motion.Motion(loop_mode=raw.loop_mode, fps=raw.fps, frames=sym.numpy().astype(np.float32))
    out.save(args.out)
    print("-"*92)
    print("saved {} frames ({:.2f}s, loop={}) -> {}".format(
        sym.shape[0], (sym.shape[0]-1)/fps, raw.loop_mode, args.out))

if __name__=="__main__":
    main()
