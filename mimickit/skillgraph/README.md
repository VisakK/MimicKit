# Yoga Skill Graph

A value-grounded, dynamically-feasible motion graph over yoga pose-skills:
train each pose as an isolated DeepMimic skill, characterise its competence with
an **initiation classifier** (not the raw critic), and connect poses with
**transition policies** trained by RSI from the source pose's terminal states
toward the target pose's competence region. Implements the loop in
`yoga_guru_spec.md` (sections 4–8, 11). Three starter poses: **handstand, crow,
scorpion** — all hand-supported, so they share a support polygon (good for
transitions).

## Why a classifier, not the critic (section 4)

A PPO critic is on-policy and over-optimistic *off-distribution* — exactly at
skill A's terminal states, which are off skill B's policy distribution. So the
handoff gate is a calibrated two-class classifier `C_B(s) -> p_init` trained from
states where B holds vs. falls. Measured on handstand: classifier **AUC 0.962**
vs raw-critic **AUC 0.824** at predicting hold-success (see `skills/handstand/
calibration.png`). The classifier operates on a **physical-state featurizer**
(`state_features.py`), deliberately NOT the policy obs (which is
reference-phase-conditioned and ill-defined off-distribution).

## On-disk world model (`skills/`)

```
skills/
  graph.yaml                 # transition matrix: nodes + directed edges + status
  feasibility.{png,json}     # zero-shot p_init matrix across all pose pairs
  <skill>/
    card.yaml                # skill card: ckpt, contact signature, hold phase, classifier, terminal stats
    dataset.pt               # labeled (features, obs, critic_value, label) for calibration
    terminal_states.pt       # quasi-static held-pose states (RSI source for transitions)
    classifier.pt            # initiation classifier C_skill
    calibration.{png,json}   # section-4 result: classifier vs raw critic
```

## Pipeline

0. **Characterise clips** (grounds the configs; no GPU):
   `tools/analyze_pose_clips.py` → per-pose contact signature + hold window +
   penetration. (handstand: hands/inverted; crow: hands/upright tuck; scorpion:
   hands/inverted backbend.)

1. **Train each pose skill** (DeepMimic-orient stack — stable kick-up + hold,
   clean critic, hand-orientation reward):
   `tools/run_yoga_skills.sh` (crow + scorpion in parallel; handstand reuses
   `output/yoga_orient/model_seed42.pt`). Configs:
   `data/envs/deepmimic_smpl_{crow,scorpion}_env.yaml`.

2. **Collect terminal states + initiation dataset** (per skill, GPU):
   ```
   mimickit/skillgraph/collect_skill.py --skill_id crow \
     --env_config data/envs/deepmimic_smpl_crow_env.yaml \
     --agent_config data/agents/deepmimic_smpl_ppo_agent.yaml \
     --model_file output/yoga_skills/crow/model.pt \
     --hold_up_z_min 0.0          # crow is upright; inverted poses use --hold_up_z_max -0.5
   ```
   A per-env **action-noise spectrum** spans B's competence boundary in one
   rollout (low-noise envs hold = positives + clean terminal states; high-noise
   envs fall = negatives).

3. **Fit classifier + section-4 calibration** (CPU):
   `mimickit/skillgraph/build_classifier.py --skill_id crow`
   → `classifier.pt`, `calibration.png/json`, GO/NO-GO (AUC ≥ 0.75 and ≥ critic).

4. **Zero-shot feasibility matrix** (CPU, learnability gate, section 11.3a):
   `mimickit/skillgraph/feasibility.py`
   → `feasibility.png`, graph edges tagged `zero_shot_candidate` /
   `train_transition` / `needs_intermediate` by `median p_init_B(S_A^term)`.

5. **Train a transition** (RSI = A's terminal states → B's pose, section 8):
   ```python
   from skillgraph import orchestrator as orch
   orch.launch_transition_training("crow", "handstand")  # writes config + launches run.py
   ```
   `envs/transition_env.py`: fixed target = B's held pose (dense DeepMimic
   shaping), per-step goal bonus `1[p_init_B(s) > θ]` paid while inside B
   (contractive — episode continues so the policy stays in B), `Goal_Frac` is the
   live scoreboard, `pose_termination` off, fall on trunk contact.

## Orchestrator tools (`orchestrator.py`, section 5)

`read/update_skill_graph`, `read_skill_card`, `query_initiation_classifier`,
`make_transition_config`, `launch_training`, `launch_transition_training`,
`monitor_training`, `collect_skill_states`. Each grounds a meta-controller action
in a real MimicKit call; an LLM can drive the section-11 loop through them.

## Status

- handstand: trained (reused), collected, classified (**GO**, AUC 0.962).
- crow, scorpion: training (200 M each); collect + classify + feasibility +
  transitions follow once trained.
