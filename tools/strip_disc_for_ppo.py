"""Convert an AMP checkpoint into a PPO-loadable one by dropping the discriminator.

An AMP model is a strict superset of a PPO model: AMPModel/AMPAgent add the
discriminator (`_model._disc_layers.*`, `_model._disc_logits.*`) and its obs
normalizer (`_disc_obs_norm.*`) on top of the shared actor + critic + obs/action
normalizers. A PPO agent's state_dict is exactly the AMP one MINUS those disc
keys, so stripping them yields a checkpoint that loads into a PPO agent under the
default strict load (which doubles as a correctness check: a strict load failure
would mean the key sets don't line up).

Used to warm-start the one-legged-crow DeepMimic(PPO) policy from the solved crow
AMP policy. See Crow_pose.MD for the source model.

Run:
  PY=/home/visakii/Documents/moves/env_isaaclab/bin/python
  $PY tools/strip_disc_for_ppo.py \
      --in output/model_yoga_amp_crow_hybrid_ft4_lowlr.pt \
      --out output/model_crow_ft4_ppo_init.pt
"""
import argparse
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_file", required=True)
    p.add_argument("--out", dest="out_file", required=True)
    p.add_argument("--keep_critic_head", action="store_true",
                   help="keep the source critic output head (default: zero it to "
                        "re-calibrate the value scale for a different reward)")
    args = p.parse_args()

    sd = torch.load(args.in_file, map_location="cpu")
    kept, dropped = type(sd)(), []
    for k, v in sd.items():
        if "disc" in k.lower():
            dropped.append(k)
        else:
            kept[k] = v

    # Reset the critic OUTPUT head to zero (keeps strict load: head keys stay
    # present). The source critic predicts the SOURCE reward's value scale (the
    # AMP crow returns ~455); under a different reward (pure DeepMimic, returns
    # ~50-180) that warm-started head is confidently wrong by ~350, and its large
    # pre-trained weights + the huge first-step gradient explode the critic (the
    # observed Critic_Loss 1.5e5 -> NaN at iter 1). Zeroing ONLY the output layer
    # makes V=0 initially (like a fresh critic, which trains stably) while keeping
    # the informative pre-trained value-trunk FEATURES; the head re-calibrates to
    # the new return scale and advantages are normalized, so the warm-started
    # ACTOR is unaffected. Trunk gradients are ~0 until the head grows, so no
    # trunk blow-up. Disable with --keep_critic_head if warm-starting within the
    # same reward.
    if (not args.keep_critic_head):
        for k in list(kept.keys()):
            if k.endswith("_critic_out.weight") or k.endswith("_critic_out.bias"):
                kept[k] = torch.zeros_like(kept[k])
                print("zeroed critic head: {}".format(k))

    print("loaded {} ({} tensors)".format(args.in_file, len(sd)))
    print("dropped {} disc tensors:".format(len(dropped)))
    for k in dropped:
        print("   - {}".format(k))
    print("kept {} tensors for PPO:".format(len(kept)))
    for k in kept:
        shp = tuple(kept[k].shape) if hasattr(kept[k], "shape") else "?"
        print("   + {:<42s} {}".format(k, shp))

    torch.save(kept, args.out_file)
    print("saved -> {}".format(args.out_file))


if __name__ == "__main__":
    main()
