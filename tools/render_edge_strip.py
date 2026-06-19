"""Render a trained transition A->B as a SKELETON MOTION-STRIP (headless, matplotlib).

Rolls the transition policy from the TRUE handoff (curriculum forced off), captures
the character's body positions over the first episode, and draws the skeleton at
~N evenly-spaced timesteps laid out left-to-right -- a readable still of the whole
A->B motion that can be shown inline (no Isaac GUI / display needed). Side view
(the sagittal plane: the horizontal axis with the largest body spread, plus up=Z),
so an inversion like crow->handstand reads clearly.

Usage: env_isaaclab/bin/python tools/render_edge_strip.py --from crow --to handstand
       [--frames 9] [--num_steps 240] [--out output/strip.png]
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_strip(frames, parents, frm, to, out, n_frames=9):
    """frames: [T,B,3] world body positions; parents: [B] parent index (-1 root)."""
    frames = np.asarray(frames)
    T = len(frames)
    # sagittal horizontal axis = the X/Y axis with the larger body spread over the clip
    spread = frames[..., :2].reshape(-1, 2).std(axis=0)
    h = 0 if spread[0] >= spread[1] else 1
    HNAME = "X" if h == 0 else "Y"
    idx = np.linspace(0, T - 1, min(n_frames, T)).round().astype(int)
    fig, ax = plt.subplots(figsize=(2.0 * len(idx), 3.8))
    cmap = plt.cm.viridis
    for k, t in enumerate(idx):
        bp = frames[t]
        cx = bp[:, h].mean()
        xs = bp[:, h] - cx + k * 1.4
        zs = bp[:, 2]
        col = cmap(k / max(len(idx) - 1, 1))
        for b in range(len(parents)):
            par = int(parents[b])
            if (par < 0 or par >= len(bp)):
                continue
            ax.plot([xs[b], xs[par]], [zs[b], zs[par]], "-", color=col, lw=2.0, alpha=0.9)
        ax.scatter(xs, zs, s=9, color=col, zorder=3)
        # mark Head (red o), Hands (black square = support), Toes (green ^) so
        # inversion is unambiguous (handstand: hands low, head just above, feet high)
        HEAD, HANDS, TOES = 13, [18, 23], [4, 8]
        if (len(bp) > 23):
            ax.scatter([xs[HEAD]], [zs[HEAD]], s=55, marker="o", facecolor="none", edgecolor="red", lw=1.6, zorder=5)
            ax.scatter(xs[HANDS], zs[HANDS], s=32, marker="s", color="black", zorder=5)
            ax.scatter(xs[TOES], zs[TOES], s=34, marker="^", color="green", zorder=5)
        ax.text(k * 1.4, min(zs.min() - 0.12, -0.05), "%.0f%%" % (100 * t / max(T - 1, 1)),
                ha="center", fontsize=8, color=col)
    ax.axhline(0.0, color="0.55", lw=1.0, ls="--")     # ground plane
    ax.set_aspect("equal"); ax.set_ylabel("up (Z), m")
    ax.set_xlabel("{} -> {}   side view ({} axis), time left->right".format(frm, to, HNAME))
    ax.set_title("{} -> {}  transition (skeleton, true handoff)".format(frm, to))
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.savefig(out, dpi=110)
    print("[strip] wrote {}".format(out))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--from", dest="frm", required=True)
    p.add_argument("--to", required=True)
    p.add_argument("--frames", type=int, default=9)
    p.add_argument("--num_steps", type=int, default=240)
    p.add_argument("--agent_config", default="data/agents/amp_smpl_transition_hybrid_agent.yaml")
    p.add_argument("--out", default=None)
    p.add_argument("--master_port", type=int, default=6981)
    args = p.parse_args()

    sys.path.insert(0, "mimickit")
    import torch
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import envs.base_env as base_env
    import util.mp_util as mp_util
    import util.util as util

    env_cfg = "data/envs/transition_{}_to_{}_env.yaml".format(args.frm, args.to)
    model = "output/yoga_transitions/{}_to_{}/model.pt".format(args.frm, args.to)
    out = args.out or "output/strip_{}_to_{}.png".format(args.frm, args.to)

    mp_util.init(0, 1, "cuda:0", args.master_port)
    util.set_rand_seed(0)
    env = env_builder.build_env(env_cfg, 1, "cuda:0", visualize=False)
    if (hasattr(env, "_curriculum_init_frac")):
        env._curriculum_init_frac = 0.0          # true handoff only
    agent = agent_builder.build_agent(args.agent_config, env, "cuda:0")
    agent.load(model)
    agent.eval(); agent.set_mode(agent._mode.__class__.TEST)

    kcm = env._kin_char_model
    parents = kcm._parent_indices.cpu().numpy()
    cid = env._get_char_id()
    NULL = base_env.DoneFlags.NULL.value

    obs, info = agent._reset_envs()
    frames = []
    for _ in range(args.num_steps):
        with torch.no_grad():
            a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        frames.append(env._engine.get_body_pos(cid)[0].cpu().numpy())
        obs, r, done, info = env.step(a)
        if (done[0].item() != NULL):
            break
    print("[strip] captured {} steps for {}->{}".format(len(frames), args.frm, args.to))
    plot_strip(np.stack(frames), parents, args.frm, args.to, out, args.frames)


if __name__ == "__main__":
    main()
