# Export a trained DQN checkpoint for deployment.
#   python python/export_model.py --ckpt models/<name>_best.pt                       # flat files next to the checkpoint
#   python python/export_model.py --ckpt X_best.pt --algo 02_dqn_cnn --name "DQN (CNN, grid)" --tryout 2 \
#          --notes ../notes/02_dqn.html                                              # publish to the model registry
# Publishing writes models/<algo>/{best.json,best.onnx,best.pt,meta.json,best.weights.js} and regenerates
# models/registry.js, which game/index.html reads (works from file://, no server). See snake_rl/registry.py.
import argparse

from snake_rl.export import ROOT, export_checkpoint


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "models" / "_runs" / "dqn_cnn_12x12_long_best.pt"))
    ap.add_argument("--algo", help="registry id, e.g. 02_dqn_cnn; omit for a flat export")
    ap.add_argument("--name", help="display name in the game's Agent dropdown")
    ap.add_argument("--tryout", type=int)
    ap.add_argument("--notes", help="link to the HTML note, relative to game/index.html")
    ap.add_argument("--description", default="")
    ap.add_argument("--run", help="results/phase2_dqn/<run> to take the training cost from (default: checkpoint name)")
    ap.add_argument("--provisional", action="store_true", help="mark as a best-so-far snapshot of a run still in training")
    ap.add_argument("--only-if-better", action="store_true", help="skip when the registered model has a better validation mean")
    a = ap.parse_args()
    export_checkpoint(a.ckpt, a.algo, a.name, a.tryout, a.notes, a.description, a.run, a.only_if_better, provisional=a.provisional)


if __name__ == "__main__":
    main()
