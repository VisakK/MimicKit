"""Evaluate the MONOLITHIC baseline on the whole plank_vinyasa trajectory from t=0.
Rolls the converged single policy from the standing start (RSI [0,0.01]) and records,
per env, how FAR along the 35s trajectory it gets before falling -> the death-time
distribution pinpoints WHICH regime a single policy can't satisfy. Captures the
longest-surviving env's rollout for a strip (vs reference) + Isaac replay states.
"""
import sys, os, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "mimickit"))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import envs.env_builder as env_builder, envs.base_env as base_env
import learning.agent_builder as agent_builder, util.mp_util as mp_util, util.util as util

D = "output/yoga_flows/plank_vinyasa/monolith_baseline"
FPS = 30
SEAMS = [(8.5, "downdog"), (14.1, "plank"), (19.1, "SIDE_PLANK"), (24.0, "plank(return)"),
         (30.3, "downdog"), (35.0, "tadasana(END)")]


def main():
    device, N = "cuda:0", 128
    mp_util.init(0, 1, device, 29793); util.set_rand_seed(7)
    env = env_builder.build_env(D + "/env_config.yaml", N, device, visualize=False)
    agent = agent_builder.build_agent(D + "/agent_config.yaml", env, device)
    agent.load(D + "/model.pt"); agent.eval(); agent.set_mode(agent._mode.__class__.TEST)
    env._init_time_range = [0.0, 0.01]
    cid = env._get_char_id(); e = env._engine
    NULL = base_env.DoneFlags.NULL.value
    max_steps = int(36.5 * FPS)

    obs, info = agent._reset_envs()
    B = e.get_body_pos(cid).shape[1]; ndof = e.get_dof_pos(cid).shape[1]
    body = np.zeros((max_steps, N, B, 3), np.float32)
    root = np.zeros((max_steps, N, 3), np.float32); rot = np.zeros((max_steps, N, 4), np.float32)
    dof = np.zeros((max_steps, N, ndof), np.float32); ref = np.zeros((max_steps, B, 3), np.float32)
    alive = torch.ones(N, dtype=torch.bool, device=device)
    death_step = torch.full((N,), max_steps, dtype=torch.long, device=device)
    last = max_steps
    for step in range(max_steps):
        with torch.no_grad():
            a = agent._a_norm.unnormalize(agent._model.eval_actor(agent._obs_norm.normalize(obs)).mode)
        body[step] = e.get_body_pos(cid).cpu().numpy(); root[step] = e.get_root_pos(cid).cpu().numpy()
        rot[step] = e.get_root_rot(cid).cpu().numpy(); dof[step] = e.get_dof_pos(cid).cpu().numpy()
        ref[step] = env._ref_body_pos[0].cpu().numpy()
        obs, r, done, info = env.step(a)
        newly = alive & (done != NULL); death_step[newly] = step; alive = alive & (done == NULL)
        if not bool(alive.any()): last = step + 1; break

    dt = (death_step.float() / FPS).cpu().numpy()
    print(f"=== MONOLITH t=0 whole-flow eval (N={N}, ref trajectory 35.4s, {last} steps rolled) ===")
    print(f"survival(s): median {np.median(dt):.1f} | mean {dt.mean():.1f} | max {dt.max():.1f} | completed>=35s: {int((dt>=35).sum())}/{N}")
    print("reached-each-milestone (still alive at that time):")
    for t, nm in SEAMS:
        print(f"   {t:5.1f}s {nm:16s}: {100*float((dt>=t).mean()):5.1f}%")
    hist, _ = np.histogram(dt, bins=[0, 8.5, 14.1, 19.1, 24.0, 30.3, 35.0, 40])
    labs = ["0->downdog", "downdog->plank", "plank->SIDEPLANK", "SIDEPLANK->plank", "plank->downdog", "downdog->tadasana", "completed(>=35s)"]
    print("where death clusters (the regime that kills it):")
    for h, lb in zip(hist, labs): print(f"   {lb:22s}: {int(h):3d} ({100*h/N:4.1f}%)")

    best = int(death_step.argmax().item()); T = int(min(death_step[best].item() + 1, last))
    print(f"longest survivor env {best}: {T/FPS:.1f}s")
    np.savez(D + "/monolith_rollout_states.npz", root_pos=root[:T, best], root_rot=rot[:T, best], dof_pos=dof[:T, best], control_freq=FPS, env_config="data/envs/monolith_plank_vinyasa_deepmimic_env.yaml")
    print(f"replay states -> {D}/monolith_rollout_states.npz ({T} frames)")

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    import xml.etree.ElementTree as ET, anim.kin_char_model as kmod
    kcm = kmod.KinCharModel("cpu"); kcm.load_char_file("data/assets/smpl/smpl_boxhands_lowtorque.xml")
    idx = {n: i for i, n in enumerate(kcm.get_body_names())}; ed = []
    w = ET.parse("data/assets/smpl/smpl_boxhands.xml").getroot().find("worldbody")
    def walk(el, pa):
        for c in el.findall("body"):
            nm = c.get("name")
            if pa in idx and nm in idx: ed.append((idx[pa], idx[nm]))
            walk(c, nm)
    for b in w.findall("body"): walk(b, b.get("name"))
    bb = body[:T, best]; rr = ref[:T]
    snaps = np.linspace(0, T - 1, 18).astype(int)
    fig, ax = plt.subplots(figsize=(26, 4))
    for kk, s in enumerate(snaps):
        xo = kk * 1.3; dv = np.linalg.svd(bb[s][:, :2] - bb[s][:, :2].mean(0))[2][0]
        for src, col, a_ in [(rr[s], "tab:green", 0.55), (bb[s], "k", 1.0)]:
            x = (src[:, :2] - bb[s][:, :2].mean(0)) @ dv + xo
            for a2, b2 in ed: ax.plot([x[a2], x[b2]], [src[a2, 2], src[b2, 2]], "-", color=col, lw=1.2, alpha=a_)
        ax.text(xo, -0.15, f"{s/FPS:.1f}s", ha="center", fontsize=7)
    ax.axhline(0, color="gray", lw=0.7); ax.set_aspect("equal")
    ax.set_title(f"MONOLITH t=0 rollout (BLACK=policy, GREEN=ref) — best env {T/FPS:.1f}s of 35.4s")
    fig.tight_layout(); fig.savefig(D + "/monolith_rollout_strip.png", dpi=95)
    print(f"strip -> {D}/monolith_rollout_strip.png")


if __name__ == "__main__":
    main()
