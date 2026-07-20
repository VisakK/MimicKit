"""Repair Router — deterministic signature->rung lookup (structured orchestration).

Companion of tools/certify_bridge.py. The certifier emits bridge_cert.json
(raw measured channels); THIS module owns the pre-registered decision logic:
  1. compute_signature(channels, thresholds) -> boolean/enum signature vector
  2. route(signature, router, history)       -> the ONE next repair rung

The rules and every threshold live in skills/repair_router.yaml so the whole
decision surface is pre-registered and auditable (the knob-freeze rule:
per-edge overrides require a logged diagnosis line, never free-form choice).
The LLM orchestrator executes the returned rung; it may free-reason ONLY when
the router returns STUCK (no rule matched, or every matching rung was already
attempted and failed) — and then must log a novelty memo proposing a new row.

Attempt history (<edge_dir>/repair_history.json, a JSON list of
{"action": ..., "outcome": "failed"|"succeeded", "note": ...}) prevents loops:
a rule whose action already FAILED on this edge is skipped, so the router
escalates instead of re-buying the same repair (the side_plank B1 lesson).

Usage:
  python tools/route_repair.py --cert <edge_dir>/bridge_cert.json \
      [--router skills/repair_router.yaml] [--history <edge_dir>/repair_history.json] \
      [--explain]
  python tools/route_repair.py --record <edge_dir>/repair_history.json \
      --action finetune_node_basin --outcome failed --note "105M, forced_takeover 0.0"
Output: route_verdict.json next to the cert; verdict printed as JSON.
"""
import argparse
import json
import os

import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ROUTER = os.path.join(REPO, "skills", "repair_router.yaml")


def compute_signature(ch, th):
    """channels (bridge_cert.json['channels']) + thresholds -> signature dict.
    Every field is derived from MEASURED numbers; nothing here is judgment."""
    sig = {}

    h = ch["harness"]
    natural_end = min(h["episode_length"], h.get("motion_length",
                                                 h["episode_length"]))
    # gate-fire certs (explicit settle_at, R2b semantics) legitimately stop
    # before the natural end — judge the harness against what was requested
    if h.get("settle_requested_s") is not None:
        natural_end = min(natural_end, h["settle_requested_s"])
    # NaN tail distance (zero survivors) cannot indict the harness — the run
    # demonstrably reached settle; reachability owns that failure
    tail = h["delivery_vs_edge_tail_pose_mean"]
    tail_ok = (tail != tail) or tail <= th["harness_tail_pose_max"]
    sig["harness_ok"] = (h["ran_to_s"] >= natural_end - 0.75
                         and h["init_at"] <= 0.001
                         and tail_ok)

    sig["node_valid"] = ch["node"]["oracle_validity_clean"] >= th["node_validity_min"]

    r = ch["reachability"]
    sig["reachable"] = (r["survive_rate_at_settle"] >= th["reachable_min_survive"]
                        and r["fell_rate_t0"] <= th["fell_max"])

    s = ch["support"]
    sig["support_correct"] = (s["req_loaded_frac"] >= th["support_req_loaded_min"]
                              and s["not_forbidden_frac"]
                              >= th["support_not_forbidden_min"])

    sig["certificate_fires"] = ch["certificate"]["arrival_rate"] >= th["certificate_min"]

    rec = ch["recovery"]["recovery_rate"]
    sig["recoverable"] = rec >= th["recover_min"]
    # judge suspect: the node demonstrably holds the delivery but the calibrated
    # certificate refuses to fire (certificate recall ~0 class -> E1, no GPU)
    sig["judge_suspect"] = (sig["recoverable"] and rec > 0
                            and ch["certificate"]["arrival_rate"]
                            < th["judge_arrival_vs_recovery_ratio"] * rec)

    p = ch["pose"]
    theta = ch["node"]["gate_theta_pose_mean"]
    sig["pose_match"] = p["pose_mean_med"] <= th["pose_theta_mult"] * theta

    st = ch["style"]
    sig["style_on_manifold"] = (
        st["max_support_z_delta_cm"] <= th["style_support_z_delta_max_cm"]
        and st["max_extension_delta_cm"] <= th["style_extension_delta_max_cm"])

    g = ch["gait"]
    sig["gait_clean"] = (g["loaded_slide_speed_mean"] <= th["gait_slide_speed_max"]
                         and g["contact_toggle_hz"] <= th["gait_contact_toggle_max_hz"])

    # balance mode from the late-window CoM-margin trace (the documented tell:
    # Mode A geometry = margin negative from the start; Mode B control = starts
    # inside then drifts out while the pose is otherwise reached)
    b = ch["balance"]
    if b["margin_neg_frac_late"] >= th["balance_geometry_neg_frac"]:
        sig["balance_mode"] = "geometry"
    elif (b["margin_start_late"] >= th["balance_control_margin_start"]
          and b["margin_end"] < 0.0):
        sig["balance_mode"] = "control"
    else:
        sig["balance_mode"] = "none"
    return sig


def _matches(rule, sig):
    for k, v in rule.get("when", {}).items():
        if sig.get(k) != v:
            return False
    return True


def route(sig, router, history=None):
    """Walk the ordered rules; first match whose action has not already FAILED
    on this edge wins. Returns the verdict dict (never raises)."""
    failed = {h["action"] for h in (history or []) if h.get("outcome") == "failed"}
    skipped = []
    for rule in router["rules"]:
        if not _matches(rule, sig):
            continue
        if rule["action"] in failed and not rule.get("terminal", False):
            skipped.append(rule["rung"])
            continue
        return dict(rung=rule["rung"], action=rule["action"],
                    target=rule.get("target"), then=rule.get("then"),
                    incident=rule.get("incident"),
                    terminal=bool(rule.get("terminal", False)),
                    skipped_attempted=skipped, signature=sig)
    return dict(rung="STUCK", action="novelty_protocol",
                target="orchestrator",
                then="No pre-registered rule matches (or all matching rungs "
                     "already failed). The LLM may now diagnose freely, but "
                     "must log a novelty memo and propose a new router row "
                     "before spending GPU.",
                incident=None, terminal=False,
                skipped_attempted=skipped, signature=sig)


def load_history(path):
    if path and os.path.exists(path):
        with open(path) as fh:
            return json.load(fh)
    return []


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cert", help="bridge_cert.json from tools/certify_bridge.py")
    p.add_argument("--router", default=DEFAULT_ROUTER)
    p.add_argument("--history", default=None,
                   help="repair_history.json (default: next to the cert)")
    p.add_argument("--explain", action="store_true",
                   help="print every rule's match result")
    p.add_argument("--record", default=None,
                   help="append an attempt to this history file and exit")
    p.add_argument("--action", default=None)
    p.add_argument("--outcome", default=None, choices=["failed", "succeeded"])
    p.add_argument("--note", default="")
    args = p.parse_args()

    if args.record:
        assert args.action and args.outcome, "--record needs --action and --outcome"
        hist = load_history(args.record)
        hist.append(dict(action=args.action, outcome=args.outcome, note=args.note))
        with open(args.record, "w") as fh:
            json.dump(hist, fh, indent=2)
        print(f"recorded {args.action}={args.outcome} -> {args.record}")
        return

    assert args.cert, "--cert required (or use --record)"
    with open(args.cert) as fh:
        cert = json.load(fh)
    with open(args.router) as fh:
        router = yaml.safe_load(fh)
    hist_path = args.history or os.path.join(os.path.dirname(args.cert),
                                             "repair_history.json")
    history = load_history(hist_path)

    sig = compute_signature(cert["channels"], router["thresholds"])
    verdict = route(sig, router, history)

    if args.explain:
        for rule in router["rules"]:
            print(f"  {'MATCH ' if _matches(rule, sig) else '      '}"
                  f"{rule['rung']:<24} when={rule.get('when')}")

    out_path = os.path.join(os.path.dirname(args.cert), "route_verdict.json")
    with open(out_path, "w") as fh:
        json.dump(verdict, fh, indent=2)
    print(json.dumps(verdict, indent=2))
    print(f"verdict -> {out_path}")


if __name__ == "__main__":
    main()
