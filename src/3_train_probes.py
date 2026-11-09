"""Train linear probes on the activations cached by stage 2 (src/2_prepare_probing_dataset.py).

For one trait and checkpoint, trains one logistic probe with an L1 penalty on the weights per layer,
submodule, pooling mode and penalty (`l1_sparsity`), on the activations of the first question set in
`sets` (default `cal`). Each probe is evaluated on a held-out part of these activations (70/30 split);
this held-out activation split is not the separate `test` question set used for steering. Probe
weights are saved under $SCRATCH/apertus-pulse/data/probes/, accuracy and ROC-AUC are logged to W&B.

Usage:
    python -m src.3_train_probes trait_id=1 model.checkpoint=apertus_v1.5_8B
"""
import os
from omegaconf import OmegaConf
import torch
import wandb
from torch.nn.functional import binary_cross_entropy_with_logits
from sklearn.metrics import accuracy_score, roc_auc_score
from hydra import main
from omegaconf import DictConfig

from src.utils.paths import ensure_scratch_env

from src.utils.data_utils import load_trait_instructions
from src.utils.model_utils import LinearProbe
from src.utils.load_utils import verify_set



def load_split(set_dir, layer, sub, pool, train_frac=0.7, cal_frac=0.0, seed=42):
    X, y = [], []
    for sign in ["pos", "neg"]:
        for p in sorted(os.listdir(os.path.join(set_dir, f"acts/layer_{layer}", sub, pool))):
            if p.endswith(".pt") and p.startswith(f"{sign}_batch_"):
                t = torch.load(os.path.join(set_dir, f"acts/layer_{layer}", sub, pool, p), map_location="cpu")
                X.append(t if isinstance(t, torch.Tensor) else torch.stack(t))
                y.append(torch.full((len(X[-1]),), 1.0 if sign == "pos" else 0.0))
    if not X:
        raise FileNotFoundError(f"No activations for layer={layer} sub={sub} pool={pool}")

    X, y = torch.cat(X), torch.cat(y)
    N = len(X)
    # Isolated generator (not torch.manual_seed) so the split does not perturb the global RNG that
    # LinearProbe's weight initialisation uses. With the same seed and the same number of samples
    # (constant across layer/submodule/pooling for one trait, checkpoint and set), every probe gets
    # the same held-out split, so layers and settings are compared on identical data.
    perm = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
    X, y = X[perm], y[perm]

    # Split into train / optional calibration / held-out parts. With cal_frac=0 (the default) there
    # is no calibration part and None is returned in its place.
    if cal_frac == 0.0:
        N_train = int(N * train_frac)
        return X[:N_train], y[:N_train], None, None, X[N_train:], y[N_train:]
    else: 
        N_train, N_cal = int(N * train_frac), int(N * cal_frac)
        return (
            X[:N_train], y[:N_train],                  # train
            X[N_train:N_train+N_cal], y[N_train:N_train+N_cal],  # cal
            X[N_train+N_cal:], y[N_train+N_cal:],      # test
        )

def train_probe(X_train: torch.Tensor, y_train: torch.Tensor, l1: float = 0.0, lr: float = 1e-3, steps: int = 2000) -> LinearProbe:
    X, y = X_train.cuda().float(), y_train.cuda().float() 
    probe = LinearProbe(X.shape[1]).cuda()
    opt = torch.optim.AdamW(probe.parameters(), lr=lr)
    
    for _ in range(steps):
        logits = probe(X).squeeze()
        loss = binary_cross_entropy_with_logits(logits, y) + l1 * probe.w.abs().sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return probe

def eval_probe(probe, X, y):
    with torch.no_grad():
        logits = probe(X.cuda().float()).cpu().sigmoid().numpy()
    y_true = y.numpy().astype(int)
    y_pred = (logits > 0.5).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, logits)
    return acc, roc_auc

# The Hydra configs interpolate ${oc.env:SCRATCH}, which OmegaConf resolves when the
# decorator below loads the config. Resolving it here first turns an unset SCRATCH into
# the clear error of utils/paths.py.
ensure_scratch_env()


@main(config_path="../configs", config_name="train_probes")
def train(cfg: DictConfig):
    clean_cfg = OmegaConf.to_container(cfg, resolve=True)

    trait_data = load_trait_instructions(cfg.trait_id)
    run_name = f"{cfg.model.checkpoint}_{trait_data['trait_name_pos']}_vs_{trait_data['trait_name_neg']}"

    set_dir = os.path.join(cfg.data_dir, "cache", cfg.model.checkpoint, f"trait_{cfg.trait_id}", cfg.sets[0])
    out_dir = os.path.join(cfg.probe_dir, cfg.model.checkpoint, f"trait_{cfg.trait_id}", cfg.sets[0], "log_probes")

    print(f"[INFO] Training probes for trait {cfg.trait_id} | checkpoint={cfg.model.checkpoint} | set={cfg.sets[0]}")
    print(f"[INFO] Activation source: {set_dir}")
    print(f"[INFO] Probes will be saved to: {out_dir}")
    
    if not verify_set(set_dir):
        print(f"[ERROR] Missing activation files in {set_dir}. Aborting.")
        return

    wandb.init(
        config=clean_cfg,
        name=run_name,
        tags=[cfg.model.checkpoint],
    )
    wandb.define_metric("layer")
    wandb.define_metric("test_acc/*", step_metric="layer")
    wandb.define_metric("test_roc_auc/*", step_metric="layer")

    results_table = wandb.Table(columns=[
        "layer", "test_acc", "test_roc_auc", "pool", "submodule", "l1", "trait_id", "model_checkpoint"
    ])
    

    layers = sorted([int(d.split("_")[-1]) for d in os.listdir(os.path.join(set_dir, "acts")) if d.startswith("layer_")])  # layers = the acts/layer_<L> folders
    
    for layer in layers:
        layer_metrics = {"layer": layer}
        for l1 in cfg.l1_sparsity:
            for sub in cfg.submodules:
                for pool in cfg.pooling_modes:
                    X_train, y_train, X_cal, y_cal, X_test, y_test = load_split(set_dir, layer, sub, pool)
                    probe = train_probe(X_train, y_train, l1=l1, lr=cfg.lr, steps=cfg.steps)
                    acc_test, roc_auc_test = eval_probe(probe, X_test, y_test)
                    results_table.add_data(layer, acc_test, roc_auc_test, pool, sub, l1, cfg.trait_id, cfg.model.checkpoint)
                    series = f"{sub}|{pool}|l1={l1}"
                    layer_metrics[f"test_acc/{series}"] = acc_test
                    layer_metrics[f"test_roc_auc/{series}"] = roc_auc_test
                    save_dir = os.path.join(out_dir, sub, pool, f"l1_{l1}")
                    os.makedirs(save_dir, exist_ok=True)
                    torch.save(probe.state_dict(), os.path.join(save_dir, f"layer_{layer}.pt"))
                    print(f"Layer {layer} {sub} {pool} (l1={l1}): test_acc={acc_test:.3f}, test_roc_auc={roc_auc_test:.3f}")
                    
                    del probe, X_train, y_train, X_cal, y_cal, X_test, y_test 
                    torch.cuda.empty_cache()
        wandb.log(layer_metrics, step=layer)

    wandb.log({"probe_results": results_table})

    wandb.finish()



if __name__ == "__main__":
    train()
