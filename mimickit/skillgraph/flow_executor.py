"""v3 Flow executor (yoga_flow_paper_plan.md §5.1).

A FLOW = an ordered list of stages through the tadasana hub. Two stage types:

  edge : roll a trained v3 EDGE policy. stage 0 RSIs at the edge's lead-in
         (init_time_range[0]); later stages INJECT the previous stage's settled
         reached-state, SE(2)-aligned to this edge's lead-in reference.
  node : DWELL — inject the previous stage's delivered state into the destination
         NODE's hold env (SE(2)-aligned to the node's hold_ref) and roll the node's
         hold policy for --dwell_seconds (pose_termination off = recovery-hold).

Each stage records the character's per-frame trajectory (body positions for the
matplotlib render + full kinematic state root/dof for the Isaac-Lab replay). The
parent stitches the arrays and produces TWO renders:
  - `<flow>.mp4`      : headless matplotlib 3D skeleton (flow_render.py), no display.
  - `<flow>_states.npz`: stitched per-frame (root, dof) for flow_replay.py, which
                         plays the flow back through the REAL Isaac-Lab renderer in a
                         visualize=True window (run on a machine with a display).

Isaac boots once per process, so each stage runs as a `--run_stage` subprocess and
the parent threads reached-state .pt files forward. The injection override and SE(2)
alignment are copied self-contained from tools/takeover_oracle.py and
tools/eval_edge_handoff.py so this module imports NO Isaac at parent scope.

Usage (repo root, env python; ONE Isaac job at a time):
  env_isaaclab/bin/python mimickit/skillgraph/flow_executor.py \
      --flow data/flows/warrior_roundtrip.yaml \
      --out_dir output/yoga_flows/warrior_roundtrip
Then, on a machine WITH a display, view the real character:
  env_isaaclab/bin/python mimickit/skillgraph/flow_replay.py \
      --states output/yoga_flows/warrior_roundtrip/warrior_roundtrip_states.npz --loops 3
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)
PY = sys.executable

STATE_KEYS = ["root_pos", "root_rot", "root_vel", "root_ang_vel", "dof_pos", "dof_vel"]


# --------------------------------------------------------------------------- #
# numpy quaternion helpers (xyzw) for the SE(2) stitch (parent scope, no Isaac)
# --------------------------------------------------------------------------- #
def _quat_rot_np(q, v):
    xyz = q[:3]
    uv = np.cross(xyz, v)
    return v + 2.0 * (q[3] * uv + np.cross(xyz, uv))


def _heading(q):
    fwd = _quat_rot_np(q, np.array([1.0, 0.0, 0.0]))
    return np.arctan2(fwd[1], fwd[0])


def _quat_z(yaw):
    return np.array([0.0, 0.0, np.sin(yaw / 2.0), np.cos(yaw / 2.0)])


def _quat_mul(a, b):
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz], axis=-1)


def _stitch(roots, rots):
    """Rigid-SE(2) chain the per-stage (root_pos[Ti,3], root_rot[Ti,4]) so each
    stage continues from the previous stage's final root (no teleport in a
    fixed-camera viewer). z untouched. dof is frame-invariant (unchanged)."""
    out_p, out_r = [], []
    prev_p = prev_h = None
    for rp, rr in zip(roots, rots):
        if prev_p is None:
            sp, sr = rp.copy(), rr.copy()
        else:
            dyaw = prev_h - _heading(rr[0])
            c, s = np.cos(dyaw), np.sin(dyaw)
            R = np.array([[c, -s], [s, c]])
            sp = rp.copy()
            sp[:, :2] = (rp[:, :2] - rp[0, :2]) @ R.T + prev_p[:2]
            qz = np.broadcast_to(_quat_z(dyaw), rr.shape)
            sr = _quat_mul(qz, rr)
        out_p.append(sp)
        out_r.append(sr)
        prev_p, prev_h = sp[-1], _heading(sr[-1])
    return np.concatenate(out_p), np.concatenate(out_r)


# --------------------------------------------------------------------------- #
# child helpers (Isaac scope)
# --------------------------------------------------------------------------- #
def _heading_align(state, ref_root_pos, ref_root_rot, torch_util):
    """SE(2)-align a batch of states into the reference frame: root xy -> ref xy,
    heading -> ref heading; z untouched. Copied from eval_edge_handoff.py."""
    q_s = torch_util.calc_heading_quat(state["root_rot"])
    q_r = torch_util.calc_heading_quat(ref_root_rot.expand_as(state["root_rot"]))
    q = torch_util.quat_mul(q_r, torch_util.quat_conjugate(q_s))
    out = {k: v.clone() for k, v in state.items()}
    out["root_rot"] = torch_util.quat_mul(q, state["root_rot"])
    for k in ("root_vel", "root_ang_vel"):
        out[k] = torch_util.quat_rotate(q, state[k])
    rp = state["root_pos"].clone()
    rp[:, 0:2] = ref_root_pos[0, 0:2]
    out["root_pos"] = rp
    return out


def _make_inject_reset(env, batch, torch):
    """Instance override of _reset_char: motion clock from stored motion_time,
    engine state from the stored batch. Copied from takeover_oracle.py."""
    import types
    cid = env._get_char_id()
    batch = {k: v.to(env._device) for k, v in batch.items()}

    def _reset_char(self, env_ids):
        n = len(env_ids)
        t = batch["motion_time"][env_ids]
        mids = torch.zeros(n, dtype=torch.long, device=self._device)
        self._motion_ids[env_ids] = mids
        self._motion_time_offsets[env_ids] = t
        rp, rr, rv, rav, jr, dv = self._motion_lib.calc_motion_frame(mids, t)
        self._ref_root_pos[env_ids] = rp
        self._ref_root_rot[env_ids] = rr
        self._ref_root_vel[env_ids] = rv
        self._ref_root_ang_vel[env_ids] = rav
        self._ref_joint_rot[env_ids] = jr
        self._ref_dof_vel[env_ids] = dv
        ref_body_pos, ref_body_rot = self._kin_char_model.forward_kinematics(
            self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot)
        self._ref_body_pos[:] = ref_body_pos
        self._ref_body_rot[:] = ref_body_rot
        self._ref_dof_pos[env_ids] = self._motion_lib.joint_rot_to_dof(jr)
        e = self._engine
        for setter, key in ((e.set_root_pos, "root_pos"), (e.set_root_rot, "root_rot"),
                            (e.set_root_vel, "root_vel"), (e.set_root_ang_vel, "root_ang_vel"),
                            (e.set_dof_pos, "dof_pos"), (e.set_dof_vel, "dof_vel")):
            setter(env_ids, cid, batch[key][env_ids].to(self._device))
        e.set_body_vel(env_ids, cid, 0.0)
        e.set_body_ang_vel(env_ids, cid, 0.0)
        if hasattr(self, "_prev_contact_force_z") and self._impulse_body_ids.shape[0] > 0:
            self._prev_contact_force_z[env_ids] = 1.0e6
        if hasattr(self, "_curr_bout_vert_impulse") and self._vert_impulse_body_ids.shape[0] > 0:
            self._curr_bout_vert_impulse[env_ids] = 0.0
            self._prev_vert_impulse_in_contact[env_ids] = True
        if self._enable_contact_obs:
            self._contact_obs_stale[env_ids] = True
        return

    return types.MethodType(_reset_char, env)


def _rollout_record(env, agent, max_steps, torch, base_env):
    """Deterministic single-env rollout. Records body positions (skeleton render)
    + full kinematic state per frame (Isaac replay). Captures the PRE-step state so
    the reached-state is the last settled pose, not a post-terminal reset frame."""
    cid = env._get_char_id()
    e = env._engine
    NULL = base_env.DoneFlags.NULL.value
    FAIL = base_env.DoneFlags.FAIL.value
    obs, info = agent._reset_envs()
    bp, root, rot, ct, dof = [], [], [], [], []
    last_state, done_type, n_steps = None, "TIME", 0
    for step in range(max_steps):
        last_state = {k: getattr(e, "get_" + k)(cid)[0:1].detach().cpu().clone()
                      for k in STATE_KEYS}
        bp.append(e.get_body_pos(cid)[0].detach().cpu().numpy())
        root.append(e.get_root_pos(cid)[0].detach().cpu().numpy())
        rot.append(e.get_root_rot(cid)[0].detach().cpu().numpy())
        dof.append(e.get_dof_pos(cid)[0].detach().cpu().numpy())
        cf = torch.linalg.vector_norm(e.get_ground_contact_forces(cid)[0], dim=-1)
        ct.append(cf.detach().cpu().numpy())
        with torch.no_grad():
            a = agent._a_norm.unnormalize(
                agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        obs, r, done, info = env.step(a)
        n_steps = step + 1
        if int(done[0]) != NULL:
            done_type = "FAIL" if int(done[0]) == FAIL else "TIME"
            break
    return bp, root, rot, ct, dof, last_state, done_type, n_steps


def run_stage(args):
    sys.path.insert(0, os.path.join(REPO, "mimickit"))
    import torch
    import yaml
    import envs.env_builder as env_builder
    import envs.base_env as base_env
    import learning.agent_builder as agent_builder
    import util.mp_util as mp_util
    import util.torch_util as torch_util
    import util.util as util

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)

    stage_dir = args.stage_dir
    agent_cfg = os.path.join(stage_dir, "agent_config.yaml")
    model = os.path.join(stage_dir, args.model)
    with open(os.path.join(stage_dir, "env_config.yaml")) as fh:
        cfg = yaml.safe_load(fh)

    if args.stage_type == "node":
        cfg["env"]["pose_termination"] = False   # recovery-hold: catch + hold
        env_cfg = args.out_traj + ".nodeenv.yaml"
        with open(env_cfg, "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        hold_ref = torch.load(os.path.join(stage_dir, "hold_states.pt"),
                              map_location=device)["hold_ref"]
        ref_rp, ref_rr = hold_ref["root_pos"].to(device), hold_ref["root_rot"].to(device)
        inj_time = hold_ref["motion_time"].to(device).reshape(1)
    else:
        env_cfg = os.path.join(stage_dir, "env_config.yaml")
        t0 = float(cfg["env"].get("init_time_range", [0.0, 0.0])[0])

    env = env_builder.build_env(env_cfg, 1, device, visualize=False)
    agent = agent_builder.build_agent(agent_cfg, env, device)
    agent.load(model)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)

    e = env._engine
    control_freq = int(round(1.0 / e.get_timestep()))
    if args.stage_type == "node":
        dur = args.dwell_seconds
    else:
        # run an edge to its NATURAL end (episode_length / motion terminal), not a
        # global cap: a truncated edge delivers an UNSETTLED pose (the warrior3 seam
        # — the flow's 12 s cap gave a shallow up_z 0.46 warrior3 vs the settled 0.14).
        dur = float(cfg["env"].get("episode_length", args.max_seconds))
    max_steps = int(round(dur * control_freq))

    if args.stage_type != "node":
        env._init_time_range = [t0, t0]   # deterministic start; bypasses t0<t1 assert

    if args.init_states and os.path.exists(args.init_states):
        blob = torch.load(args.init_states, map_location=device)
        state = {k: blob[k].to(device) for k in STATE_KEYS}
        if args.stage_type == "node":
            aligned = _heading_align(state, ref_rp, ref_rr, torch_util)
            aligned["motion_time"] = inj_time
        else:
            mids0 = torch.zeros(1, dtype=torch.long, device=device)
            t0t = torch.full((1,), t0, device=device)
            rp, rr, _, _, _, _ = env._motion_lib.calc_motion_frame(mids0, t0t)
            aligned = _heading_align(state, rp, rr, torch_util)
            aligned["motion_time"] = t0t
        # damp residual transit momentum on handoff (a hold-node dwell should
        # catch a SETTLED arrival; an edge stage keeps velocities for the transit)
        if args.handoff_vel_scale != 1.0:
            for vk in ("root_vel", "root_ang_vel", "dof_vel"):
                aligned[vk] = aligned[vk] * args.handoff_vel_scale
        env._reset_char = _make_inject_reset(env, aligned, torch)

    kcm = env._kin_char_model
    bp, root, rot, ct, dof, last_state, done_type, n_steps = _rollout_record(
        env, agent, max_steps, torch, base_env)

    torch.save({k: last_state[k] for k in STATE_KEYS}, args.out_reached)
    np.savez(args.out_traj,
             body_pos=np.stack(bp), root_pos=np.stack(root),
             root_rot=np.stack(rot), dof_pos=np.stack(dof), contact=np.stack(ct),
             parents=kcm._parent_indices.detach().cpu().numpy(),
             body_names=np.array(kcm.get_body_names()),
             control_freq=control_freq, done_type=done_type,
             n_steps=n_steps, edge=args.label, stage_type=args.stage_type,
             env_config=os.path.join(stage_dir, "env_config.yaml"))
    print(f"[stage:{args.label}({args.stage_type})] {n_steps} steps "
          f"({n_steps / control_freq:.2f}s), done={done_type}", flush=True)


# --------------------------------------------------------------------------- #
# flow resolution (from a flow yaml, or from the v3 graph via --flow_id)
# --------------------------------------------------------------------------- #
def resolve_flow_from_graph(flow_id, graph_path):
    """Expand a graph flow's waypoints -> executor stages: consecutive nodes -> the
    edge between them (from the graph); a waypoint with dwell_s>0 -> a node-hold stage
    after arriving. Missing/not-on-disk edges are flagged for the orchestrator."""
    import yaml
    g = yaml.safe_load(open(graph_path))
    flow = next((f for f in g.get("flows", []) if f["id"] == flow_id), None)
    if flow is None:
        raise SystemExit(f"flow '{flow_id}' not found in {graph_path}")
    edges = {(e["from"], e["to"]): e for e in g["edges"]}
    nodes = {n["id"]: n for n in g["nodes"]}
    wps = flow["waypoints"]
    stages, missing = [], []
    for a, b in zip(wps, wps[1:]):
        na, nb = a["node"], b["node"]
        e = edges.get((na, nb))
        if e is None:
            missing.append(f"{na}->{nb}: NO EDGE in graph")
            stages.append({"type": "edge", "label": f"{na}_to_{nb}", "dir": None,
                           "model": None, "on_disk": False})
        else:
            od = bool(e.get("on_disk"))
            if not od:
                missing.append(f"{na}->{nb}: edge '{e['id']}' status={e.get('status')} (not on disk)")
            stages.append({"type": "edge", "label": e["id"], "dir": e.get("dir"),
                           "model": e.get("model") or "model.pt", "on_disk": od})
        if b.get("dwell_s", 0) > 0:
            n = nodes.get(nb)
            od = bool(n and n.get("on_disk"))
            if not od:
                missing.append(f"dwell {nb}: node not on disk")
            stages.append({"type": "node", "label": nb + "_hold",
                           "dir": n["dir"] if n else None,
                           "model": (n.get("model", "model.pt") if n else "model.pt"),
                           "dwell_seconds": b["dwell_s"], "on_disk": od})
    return flow_id, stages, missing


def load_stages(args):
    import yaml
    if args.flow_id:
        return resolve_flow_from_graph(args.flow_id, args.graph)
    flow = yaml.safe_load(open(args.flow))
    name = flow.get("name", os.path.splitext(os.path.basename(args.flow))[0])
    stages = []
    for st in flow["stages"]:
        stype = st.get("type", "edge")
        label = st.get("edge") or st.get("node") or os.path.basename(st["dir"])
        model = st.get("model", "model.pt")
        s = {"type": stype, "label": label, "dir": st["dir"], "model": model,
             "on_disk": os.path.exists(os.path.join(st["dir"], model))}
        for k in ("dwell_seconds", "max_seconds", "handoff_vel_scale"):
            if k in st:
                s[k] = st[k]
        stages.append(s)
    return name, stages, []


# --------------------------------------------------------------------------- #
# parent: orchestrate the flow (no Isaac; subprocess per stage; render)
# --------------------------------------------------------------------------- #
def orchestrate(args):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import flow_render

    name, stages, missing = load_stages(args)
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    print(f"[flow:{name}] resolved {len(stages)} stages -> {out_dir}", flush=True)
    for st in stages:
        tail = (f"dwell {st.get('dwell_seconds')}s" if st["type"] == "node" else str(st["dir"]))
        print(f"    - {st['type']:5s} {st['label']:30s} on_disk={st['on_disk']!s:5s} {tail}")
    if missing:
        print(f"[flow:{name}] MISSING ({len(missing)}) — the orchestrator must build these:")
        for m in missing:
            print("    -", m)
    if args.dry_run:
        plan = {"flow": name, "missing": missing,
                "stages": [{k: st[k] for k in ("type", "label", "on_disk")} for st in stages]}
        with open(os.path.join(out_dir, "flow_plan.json"), "w") as fh:
            json.dump(plan, fh, indent=2)
        print(f"[flow:{name}] --dry_run: wrote plan, not running.")
        return
    # run the runnable prefix; stop at the first not-on-disk stage
    runnable = []
    for st in stages:
        if not st["on_disk"]:
            msg = f"stage '{st['label']}' not on disk"
            if args.allow_partial:
                print(f"[flow:{name}] stopping before {msg} (--allow_partial)")
                break
            print(f"[flow:{name}] ABORT: {msg}. Build it, or pass --allow_partial to run the prefix.")
            return
        runnable.append(st)
    stages = runnable

    reached = ""
    report = {"flow": name, "missing": missing, "stages": []}
    for i, st in enumerate(stages):
        stype, label = st["type"], st["label"]
        out_reached = os.path.join(out_dir, f"stage{i}_reached.pt")
        out_traj = os.path.join(out_dir, f"stage{i}_traj.npz")
        cmd = [PY, os.path.abspath(__file__), "--run_stage",
               "--stage_type", stype, "--stage_dir", st["dir"],
               "--model", st.get("model", "model.pt"), "--label", label,
               "--init_states", reached, "--out_reached", out_reached,
               "--out_traj", out_traj,
               "--max_seconds", str(st.get("max_seconds", args.max_seconds)),
               "--dwell_seconds", str(st.get("dwell_seconds", args.dwell_seconds)),
               "--handoff_vel_scale",
               str(st.get("handoff_vel_scale", 0.0 if stype == "node" else 1.0)),
               "--master_port", str(args.master_port + i),
               "--rand_seed", str(args.rand_seed), "--device", args.device]
        print(f"[flow:{name}] stage {i}: {label} ({stype})", flush=True)
        r = subprocess.run(cmd, cwd=REPO)
        if r.returncode != 0:
            print(f"[flow:{name}] stage {i} ({label}) crashed (rc={r.returncode}); stop.")
            report["stages"].append({"i": i, "label": label, "type": stype, "status": "crashed"})
            break
        meta = np.load(out_traj, allow_pickle=True)
        report["stages"].append({"i": i, "label": label, "type": stype,
                                 "done": str(meta["done_type"]), "n_steps": int(meta["n_steps"])})
        print(f"[flow:{name}] stage {i} done={str(meta['done_type'])} ({int(meta['n_steps'])} steps)")
        reached = out_reached

    done_stages = [s for s in report["stages"] if "n_steps" in s]
    if done_stages:
        bp, ct = [], []
        roots, rots, dofs = [], [], []
        bounds, cursor = [], 0
        parents = control_freq = env_config = None
        for s in done_stages:
            d = np.load(os.path.join(out_dir, f"stage{s['i']}_traj.npz"), allow_pickle=True)
            T = d["body_pos"].shape[0]
            bp.append(d["body_pos"]); ct.append(d["contact"])
            roots.append(d["root_pos"]); rots.append(d["root_rot"]); dofs.append(d["dof_pos"])
            bounds.append((cursor, cursor + T, s["label"])); cursor += T
            parents = d["parents"]; control_freq = int(d["control_freq"])
            env_config = str(d["env_config"])
        # 1) headless matplotlib skeleton mp4 (root-tracking camera, no display)
        vid = flow_render.render_flow(
            np.concatenate(bp), parents, np.concatenate(ct),
            np.concatenate(roots), np.concatenate(rots), control_freq,
            os.path.join(out_dir, f"{name}.mp4"), stage_bounds=bounds, title=name)
        report["video"] = vid
        # 2) stitched kinematic states for the Isaac-Lab replay (real mesh char)
        s_root, s_rot = _stitch(roots, rots)
        states_path = os.path.join(out_dir, f"{name}_states.npz")
        np.savez(states_path, root_pos=s_root, root_rot=s_rot,
                 dof_pos=np.concatenate(dofs), control_freq=control_freq,
                 env_config=env_config, stage_bounds=np.array(bounds, dtype=object))
        report["states"] = states_path
        report["total_frames"] = cursor
        print(f"[flow:{name}] Isaac replay states -> {states_path}")

    with open(os.path.join(out_dir, "flow_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run_stage", action="store_true", help="child: one stage")
    p.add_argument("--flow", help="flow yaml (stages: [{type,edge|node,dir,model}])")
    p.add_argument("--flow_id", help="resolve a flow from the v3 graph (skills/graph.yaml) by id")
    p.add_argument("--graph", default="skills/graph.yaml")
    p.add_argument("--dry_run", action="store_true", help="resolve + print the plan; don't run")
    p.add_argument("--allow_partial", action="store_true",
                   help="run the runnable prefix, stopping before the first missing stage")
    p.add_argument("--out_dir", default="output/yoga_flows/flow")
    p.add_argument("--stage_type", default="edge", choices=["edge", "node"])
    p.add_argument("--stage_dir")
    p.add_argument("--model", default="model.pt")
    p.add_argument("--label", default="stage")
    p.add_argument("--init_states", default="")
    p.add_argument("--out_reached")
    p.add_argument("--out_traj")
    p.add_argument("--max_seconds", type=float, default=12.0)
    p.add_argument("--dwell_seconds", type=float, default=4.0)
    p.add_argument("--handoff_vel_scale", type=float, default=1.0,
                   help="scale injected velocities (node dwell -> 0 = settle-and-hold)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29780)
    p.add_argument("--rand_seed", type=int, default=42)
    args = p.parse_args()
    if args.run_stage:
        run_stage(args)
    else:
        orchestrate(args)


if __name__ == "__main__":
    main()
