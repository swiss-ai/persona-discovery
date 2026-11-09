"""Check for cached activations before probe training (src/3_train_probes.py)."""
import os


def verify_set(set_dir):
    """Return True if `set_dir` has the expected activation folders with both batch types.

    Checks that `<set_dir>/acts/` exists and that every <layer>/<submodule>/<pooling> folder found
    under it contains at least one `pos_batch*` and one `neg_batch*` file. It does not check that all
    layers or batches are present or that the files can be loaded.
    """
    acts_dir = os.path.join(set_dir, "acts")
    if not os.path.exists(acts_dir):
        print(f"[DEBUG] MISSING acts dir in {set_dir}")
        return False
    for layer in sorted(os.listdir(acts_dir)):
        ldir = os.path.join(acts_dir, layer)
        for sub in sorted(os.listdir(ldir)):
            sdir = os.path.join(ldir, sub)
            for pool in sorted(os.listdir(sdir)):
                pdir = os.path.join(sdir, pool)
                files = os.listdir(pdir)
                if not any(f.startswith("pos_batch") for f in files):
                    print(f"[DEBUG] MISSING pos in {layer}/{sub}/{pool}")
                    return False
                if not any(f.startswith("neg_batch") for f in files):
                    print(f"[DEBUG] MISSING neg in {layer}/{sub}/{pool}")
                    return False
    print(f"[DEBUG] OK: all activations present in {set_dir}")
    return True
