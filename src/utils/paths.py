"""Path helpers that do not depend on the current working directory.

Hydra may run a stage inside its own output folder (depending on `version_base` and
`hydra.job.chdir`), so the stages build paths with these helpers instead of relative paths:

  repo_path(...)  files in the checkout: configs/, data/traits/, data/plots/. Anchored to this file.
  data_path(...)  large, regenerable data outside the checkout, under $SCRATCH/apertus-pulse/data:
                  questions, cached activations, probes, steered completions, judge results.

Resolving the data root requires the SCRATCH environment variable (the Hydra configs use it too, via
${oc.env:SCRATCH}); if it is not set, scratch_root() raises an error instead of guessing a location.
repo_path() works without it.
"""
import os

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def repo_path(*parts) -> str:
    return os.path.join(_REPO_ROOT, *parts)


def scratch_root() -> str:
    env = os.environ.get("SCRATCH")
    if env:
        return os.path.abspath(os.path.expanduser(env))

    raise RuntimeError(
        "Cannot resolve the data directory: the SCRATCH environment variable is not set.\n"
        "  Set it to a directory with enough space; the data lives in $SCRATCH/apertus-pulse/data:\n"
        "    export SCRATCH=/path/to/scratch"
    )


def data_root() -> str:
    return os.path.join(scratch_root(), "apertus-pulse", "data")


def data_path(*parts) -> str:
    return os.path.join(data_root(), *parts)


def ensure_scratch_env() -> str:
    """Resolve $SCRATCH and export it, for the Hydra stages.

    The Hydra configs interpolate `${oc.env:SCRATCH}`, which OmegaConf resolves at config-load
    time. Calling this at import time in a Hydra entry point turns an unset SCRATCH into the clear
    error above (instead of an OmegaConf interpolation error) and exports the normalised path.
    """
    root = scratch_root()
    os.environ["SCRATCH"] = root
    return root
