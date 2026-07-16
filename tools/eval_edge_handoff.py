"""Edge handoff evaluation (Yoga_edge_framework_v3.md §3.2 use-1).

Stage 1 (this process, edge env boot): roll the trained EDGE policy
deterministically from the true handoff start (RSI at the lead-in), evaluate
node B's calibrated ARRIVAL CERTIFICATE (<B>/gate.yaml) on the live sim state
every step, and record the full state at the first DWELL-COMPLETE event per
episode. Arrival states are SE(2)-aligned into B's reference frame (these
envs track the global root, so cross-clip heading/xy must be reconciled; z is
already shared — both clips are ground-baked).

Stage 2 (subprocess — Isaac boots once per process): tools/takeover_oracle.py
injects the aligned arrival states into B's node env and runs B's policy for
5 s. Edge success = arrival-rate x takeover pass-rate.

Usage (repo root, env python; ONE Isaac job at a time):
  env_isaaclab/bin/python tools/eval_edge_handoff.py \
      --edge_env data/envs/edge_crow_to_tadasana_env.yaml \
      --edge_agent data/agents/amp_smpl_transition_lowlr_agent.yaml \
      --edge_model output/yoga_edges_v3/crow_to_tadasana/model.pt \
      --target_node_dir output/yoga_nodes_v2/tadasana_lt_ampft \
      --out_dir output/yoga_edges_v3/crow_to_tadasana
Outputs: <out_dir>/arrival_states.pt (+ .oracle.pt), handoff_report.json
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np
import torch
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "mimickit"))
os.chdir(REPO)

import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.torch_util as torch_util
import util.util as util

sys.path.insert(0, os.path.join(REPO, "mimickit", "skillgraph"))
import state_features as sf

PY = sys.executable


def heading_align(state, ref_root_pos, ref_root_rot):
    """SE(2)-align a batch of states into the reference frame: root xy ->
    ref xy, heading -> ref heading; z untouched (shared ground datum)."""
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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--edge_env", required=True)
    p.add_argument("--edge_agent", required=True)
    p.add_argument("--edge_model", required=True)
    p.add_argument("--target_node_dir", required=True)
    p.add_argument("--num_envs", type=int, default=128)
    p.add_argument("--max_seconds", type=float, default=8.0)
    p.add_argument("--init_at", type=float, default=0.0,
                   help="RSI time (s) — 0 = true handoff from the lead-in")
    p.add_argument("--forced_handoff_at", type=float, default=6.0,
                   help="s; also oracle the surviving envs' states at this time "
                        "even if the certificate never fired (basin ground truth)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--master_port", type=int, default=29630)
    p.add_argument("--rand_seed", type=int, default=42)
    p.add_argument("--out_dir", required=True)
    args = p.parse_args()

    B = args.target_node_dir
    gate = yaml.safe_load(open(os.path.join(B, "gate.yaml")))
    b_blob = torch.load(os.path.join(B, "hold_states.pt"), map_location="cpu")
    hold_ref = b_blob["hold_ref"]
    b_meta = b_blob["meta"]
    b_char_weight = b_meta["char_weight"]
    os.makedirs(args.out_dir, exist_ok=True)

    device = args.device
    mp_util.init(0, 1, device, args.master_port)
    util.set_rand_seed(args.rand_seed)
    env = env_builder.build_env(args.edge_env, args.num_envs, device, visualize=False)
    agent = agent_builder.build_agent(args.edge_agent, env, device)
    agent.load(args.edge_model)
    agent.eval()
    agent.set_mode(agent._mode.__class__.TEST)

    env._init_time_range = [args.init_at, args.init_at]   # exam start, no spread

    kcm = env._kin_char_model
    body_names = kcm.get_body_names()
    contact_ids = env._obs_contact_body_ids
    com_w = env._com_body_weights
    control_freq = int(round(1.0 / env._engine.get_timestep()))
    dwell_steps = int(round(float(gate["dwell_s"]) * control_freq))
    max_steps = int(round(args.max_seconds * control_freq))
    N = args.num_envs
    NULL, FAIL = base_env.DoneFlags.NULL.value, base_env.DoneFlags.FAIL.value

    # gate pieces (unit indices resolved against the edge env's contact list)
    sys.path.insert(0, os.path.join(REPO, "tools"))
    import calibrate_gate as cg
    ref_bp = hold_ref["body_pos"].to(device)
    ref_rr = hold_ref["root_rot"].to(device)
    hold_up_z = float(hold_ref["up_z"][0])
    unit_of = cg.UNIT_OF
    units = {}
    for i in contact_ids.tolist():
        units.setdefault(unit_of.get(body_names[i], body_names[i]), []).append(i)
    unit_names = sorted(units)
    req_idx = [unit_names.index(u) for u in gate["required_contacts"] if u in unit_names]
    forb_idx = [unit_names.index(u) for u in gate["forbidden_contacts"] if u in unit_names]
    alpha_mg = gate["load_share_alpha"] * b_char_weight

    cid = env._get_char_id()
    e = env._engine

    def gate_pass():
        bp = e.get_body_pos(cid); rr = e.get_root_rot(cid)
        cur_rel = bp[:, 1:, :] - bp[:, 0:1, :]
        import envs.char_env as char_env
        cur_local = char_env.convert_to_local_body_pos(rr, cur_rel)
        ref_rel = ref_bp[:, 1:, :] - ref_bp[:, 0:1, :]
        ref_local = char_env.convert_to_local_body_pos(ref_rr, ref_rel)
        d = torch.linalg.vector_norm(cur_local - ref_local, dim=-1)
        pd_mean, pd_max = d.mean(dim=-1), d.max(dim=-1)[0]
        f = torch.linalg.vector_norm(e.get_ground_contact_forces(cid), dim=-1)
        uf = torch.stack([f[:, units[u]].sum(dim=-1) for u in unit_names], dim=-1)
        req = (uf[:, req_idx] > cg.REQ_N).all(dim=-1) if req_idx else \
            torch.ones(N, dtype=torch.bool, device=device)
        sup_idx = [i for i, u in enumerate(unit_names) if u not in gate["forbidden_contacts"]]
        share = uf[:, sup_idx].sum(dim=-1)
        forb = (uf[:, forb_idx] > cg.FORBID_N).any(dim=-1) if forb_idx else \
            torch.zeros(N, dtype=torch.bool, device=device)
        up_l = torch.zeros(N, 3, device=device); up_l[:, 2] = 1.0
        upz = torch_util.quat_rotate(rr, up_l)[:, 2]
        bv = e.get_body_vel(cid)
        com_v = torch.einsum("nbk,b->nk", bv, com_w)
        com_speed = torch.linalg.vector_norm(com_v[:, :2], dim=-1)
        dof_speed = e.get_dof_vel(cid).abs().mean(dim=-1)
        feats = sf.compute_state_features(
            root_pos=e.get_root_pos(cid), root_rot=rr, root_vel=e.get_root_vel(cid),
            root_ang_vel=e.get_root_ang_vel(cid), body_pos=bp,
            contact_forces=e.get_ground_contact_forces(cid),
            contact_body_ids=contact_ids, key_body_ids=env._key_body_ids,
            com_weights=com_w,
            force_threshold=getattr(env, "_obs_contact_force_threshold", 1.0),
            support_dirs=getattr(env, "_support_polygon_dirs", None))
        margin = feats[:, 7]
        chans = dict(
            pose_mean=pd_mean < gate["theta_pose_mean"],
            pose_max=pd_max < gate["theta_pose_max"],
            req_loaded=req, load_share=share >= alpha_mg, not_forbidden=~forb,
            up_z=(upz - hold_up_z).abs() < gate["up_z_tol"],
            margin=margin > gate["com_margin_min"],
            com_speed=com_speed < gate["com_speed_max"],
            dof_speed=dof_speed < gate["dof_speed_max"])
        ok = torch.ones(N, dtype=torch.bool, device=device)
        for v in chans.values():
            ok &= v
        return ok, pd_mean, chans

    obs, info = agent._reset_envs()
    dwell = torch.zeros(N, dtype=torch.long, device=device)
    arrived = torch.zeros(N, dtype=torch.bool, device=device)
    fell = torch.zeros(N, dtype=torch.bool, device=device)
    ended = torch.zeros(N, dtype=torch.bool, device=device)
    arr_pd = torch.full((N,), float("nan"), device=device)
    arr_state = None
    strip_frames = []   # body_pos of env 0 (+ its ref) for the sign-off strip
    ch_acc = {}         # per-channel pass fraction over the late window
    forced_state = None # states of surviving envs at --forced_handoff_at (the
                        # oracle-only fallback: is the DELIVERED state in B's
                        # true basin even if the certificate refuses it?)
    forced_step = int(round(args.forced_handoff_at * control_freq))

    for step in range(max_steps):
        with torch.no_grad():
            a = agent._a_norm.unnormalize(
                agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        obs, r, done, info = env.step(a)
        if step % 5 == 0 and not bool(ended[0]):
            strip_frames.append((step, e.get_body_pos(cid)[0].cpu().numpy(),
                                 env._ref_body_pos[0].cpu().numpy()))
        ok, pd_mean, chans = gate_pass()
        # channel diagnostics over the LATE window (post-transition; where an
        # arrival should be certifiable)
        if step >= int(0.6 * max_steps):
            live_now = (~ended).float()
            denom = float(live_now.sum())
            if denom > 0:
                for k, v in chans.items():
                    ch_acc[k] = ch_acc.get(k, 0.0) + float((v.float() * live_now).sum()) / denom
                ch_acc["_n"] = ch_acc.get("_n", 0) + 1
        live = ~arrived & ~ended
        dwell = torch.where(ok & live, dwell + 1, torch.zeros_like(dwell))
        just = (dwell >= dwell_steps) & live
        if bool(just.any()):
            st = {k: getattr(e, "get_" + k)(cid).clone() for k in
                  ["root_pos", "root_rot", "root_vel", "root_ang_vel",
                   "dof_pos", "dof_vel"]}
            if arr_state is None:
                arr_state = {k: torch.zeros_like(v) for k, v in st.items()}
            for k in arr_state:
                arr_state[k][just] = st[k][just]
            arr_pd[just] = pd_mean[just]
            arrived |= just
        if step == forced_step:
            alive = ~ended
            if bool(alive.any()):
                forced_state = ({k: getattr(e, "get_" + k)(cid).clone() for k in
                                 ["root_pos", "root_rot", "root_vel",
                                  "root_ang_vel", "dof_pos", "dof_vel"]}, alive.clone())
        fell |= (done == FAIL) & ~arrived
        ended |= (done != NULL)
        if bool((arrived | ended).all()) and step > forced_step:
            break

    n_arr = int(arrived.sum())
    arrival_rate = n_arr / N
    print(f"[handoff:{os.path.basename(args.out_dir)}] arrival (dwell-complete) "
          f"{arrival_rate:.3f} ({n_arr}/{N}); fell-before-arrival "
          f"{float(fell.float().mean()):.3f}")

    report = dict(edge=os.path.basename(args.out_dir),
                  target=os.path.basename(B), n_episodes=N,
                  arrival_rate=round(arrival_rate, 4),
                  fell_rate=round(float(fell.float().mean()), 4),
                  dwell_s=float(gate["dwell_s"]), init_at=args.init_at)
    n_acc = ch_acc.pop("_n", 0)
    if n_acc:
        report["late_window_channel_pass"] = {k: round(v / n_acc, 3)
                                              for k, v in ch_acc.items()}
        print("late-window per-channel pass:", report["late_window_channel_pass"])

    if n_arr > 0:
        idx = arrived.nonzero(as_tuple=False).flatten()
        state = {k: v[idx].cpu() for k, v in arr_state.items()}
        aligned = heading_align(state, hold_ref["root_pos"], hold_ref["root_rot"])
        aligned["motion_time"] = hold_ref["motion_time"].expand(len(idx)).clone()
        aligned["arrival_pose_dist"] = arr_pd[idx].cpu()
        blob_path = os.path.join(args.out_dir, "arrival_states.pt")
        torch.save(aligned, blob_path)
        with open(os.path.join(args.out_dir, "handoff_report.json"), "w") as fh:
            json.dump(report, fh, indent=2)
        # stage 2: takeover oracle in a fresh process (Isaac boots once/process)
        print("[handoff] stage 2: takeover oracle on the arrival states ...", flush=True)
        r2 = subprocess.run(
            [PY, "tools/takeover_oracle.py", "--node_dir", B,
             "--states", blob_path, "--master_port", str(args.master_port + 1)],
            capture_output=True, text=True)
        print("\n".join(l for l in r2.stdout.splitlines() if "oracle" in l or "pass" in l))
        if r2.returncode != 0:
            print(r2.stderr[-2000:])
            raise SystemExit("takeover oracle failed")
        orc = torch.load(blob_path + ".oracle.pt", map_location="cpu")
        takeover = float(orc["verdict"].float().mean())
        report["takeover_pass_rate"] = round(takeover, 4)
        report["edge_success"] = round(arrival_rate * takeover, 4)
        # worst-k arrivals by pose_dist for the rendered sign-off
        k = min(6, len(idx))
        worst = torch.topk(aligned["arrival_pose_dist"], k).indices.tolist()
        report["worst_arrival_pose_dists"] = [round(float(aligned["arrival_pose_dist"][i]), 4)
                                              for i in worst]
    # forced-handoff oracle: is the delivered state in B's TRUE basin, even if
    # the certificate refused it? (basin ground truth vs certificate recall)
    if forced_state is not None:
        st, alive = forced_state
        idx = alive.nonzero(as_tuple=False).flatten()
        state = {k: v[idx].cpu() for k, v in st.items()}
        aligned = heading_align(state, hold_ref["root_pos"], hold_ref["root_rot"])
        aligned["motion_time"] = hold_ref["motion_time"].expand(len(idx)).clone()
        fpath = os.path.join(args.out_dir, "forced_arrival_states.pt")
        torch.save(aligned, fpath)
        print(f"[handoff] forced-handoff oracle on {len(idx)} surviving envs "
              f"@t={args.forced_handoff_at}s ...", flush=True)
        r3 = subprocess.run(
            [PY, "tools/takeover_oracle.py", "--node_dir", B,
             "--states", fpath, "--master_port", str(args.master_port + 2)],
            capture_output=True, text=True)
        if r3.returncode == 0:
            orc3 = torch.load(fpath + ".oracle.pt", map_location="cpu")
            report["forced_handoff_survivors"] = int(len(idx))
            report["forced_takeover_pass_rate"] = round(float(orc3["verdict"].float().mean()), 4)
        else:
            print(r3.stderr[-1500:])

    with open(os.path.join(args.out_dir, "handoff_report.json"), "w") as fh:
        json.dump(report, fh, indent=2)
    print(json.dumps(report, indent=2))

    # sign-off strip: env-0 rollout, policy (blue/black) vs reference (green)
    if strip_frames:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import xml.etree.ElementTree as ET
        idx_map = {n: i for i, n in enumerate(body_names)}
        edges = []
        world = ET.parse("data/assets/smpl/smpl_boxhands.xml").getroot().find("worldbody")
        def walk(elem, parent):
            for child in elem.findall("body"):
                nm = child.get("name")
                if parent in idx_map and nm in idx_map:
                    edges.append((idx_map[parent], idx_map[nm]))
                walk(child, nm)
        for b in world.findall("body"):
            walk(b, b.get("name"))
        take = strip_frames[::max(1, len(strip_frames) // 10)][:10]
        fig, ax = plt.subplots(figsize=(2.0 * len(take), 4))
        for k, (step, bp, rp) in enumerate(take):
            span = bp[:, :2] - bp[:, :2].mean(0)
            dirv = np.linalg.svd(span)[2][0]
            x = span @ dirv + k * 1.5
            xr = (rp[:, :2] - bp[:, :2].mean(0)) @ dirv + k * 1.5
            for a2, b2 in edges:
                ax.plot([xr[a2], xr[b2]], [rp[a2, 2], rp[b2, 2]], "-",
                        color="tab:green", lw=1.0, alpha=0.6)
                ax.plot([x[a2], x[b2]], [bp[a2, 2], bp[b2, 2]], "k-", lw=1.3)
            ax.text(k * 1.5, -0.15, f"{step/control_freq:.1f}s", ha="center", fontsize=8)
        ax.axhline(0, color="gray", lw=0.8)
        ax.set_aspect("equal")
        ax.set_title(f"{os.path.basename(args.out_dir)} rollout (black=policy, green=reference)")
        fig.tight_layout()
        strip_path = os.path.join(args.out_dir, "handoff_strip.png")
        fig.savefig(strip_path, dpi=110)
        print(f"strip -> {strip_path}")


if __name__ == "__main__":
    main()
