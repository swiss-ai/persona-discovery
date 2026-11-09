# persona-discovery

This repository contains the code of the bachelor's thesis *Active Discovery of Persona Representations in Large
Language Models* (Wanja Pletscher, ETH Zürich, IVIA lab, supervised by Anna Hedström, 2026).

For thirty counselling traits from a competency taxonomy, the code trains linear probes on the activations of
open-weight Apertus models to separate responses generated under instructions to express a trait from responses
expressing its opposite. It analyses the geometry of the resulting trait directions, uses them for activation
steering, and has an LLM judge score which traits the steered responses express. The central question is whether
geometric similarity between trait directions predicts cross-trait steering effects.

The experiments use three models:

| Checkpoint key | Model |
|---|---|
| `apertus_v1.5_8B` | [swiss-ai/Apertus-v1.5-8B](https://huggingface.co/swiss-ai/Apertus-v1.5-8B) |
| `apertus_mini_5B_base` | [swiss-ai/Apertus-v1.1-0.5B](https://huggingface.co/swiss-ai/Apertus-v1.1-0.5B) |
| `apertus_mini_5B_instruct` | [swiss-ai/Apertus-v1.1-0.5B-Instruct](https://huggingface.co/swiss-ai/Apertus-v1.1-0.5B-Instruct) |

## Installation

Python 3.12 or newer. From the repository root:

```bash
pip install -r requirements-analysis.txt   # analysis and figure stages
pip install ".[pipeline]"                  # data-preparation, probing, steering and judge stages
```

[`requirements-analysis.txt`](requirements-analysis.txt) pins the exact package versions with which the thesis
figures were verified. The second command installs the dependencies declared in
[`pyproject.toml`](pyproject.toml) together with its `pipeline` extra, which adds the packages of steps 1, 3, 4, 6
and the judge below, including W&B and the [swiss-ai/transformers](https://github.com/swiss-ai/transformers) revision
used for the thesis runs. The stages are run from the repository root, so the project itself does not need to be
installed. The figure scripts convert SVG to PNG and PDF with `rsvg-convert` (librsvg) or, if that is missing,
ImageMagick.

## Data

The trait definitions and their instruction pairs are included in [`data/traits/`](data/traits/). The thesis results
(probes, steered responses and judge scores) are not public yet, so the later steps need the earlier ones to have been
run first.

Large data lives under `$SCRATCH/apertus-pulse/data`. Set `SCRATCH` before running a stage that uses it; each stage
creates its own output folders there. Figures and analysis results are written to `data/plots/`.

```bash
export SCRATCH=/path/to/scratch
```

## Pipeline

The scripts in [`src/`](src/) are numbered in the order in which data flows through them and run from the repository
root as `python -m src.<script>`. Stages 2, 3, 5a, 5b, 6a and 6b are configured with [Hydra](https://hydra.cc): their
defaults are in [`configs/`](configs/) (models in [`configs/checkpoints.yaml`](configs/checkpoints.yaml)) and are
overridden as `key=value`. The other scripts take command-line options, described at the top of each script and with
`--help`. The settings of the thesis runs are listed in the appendix of the thesis.

The examples below follow one model, `apertus_v1.5_8B`, one trait, 7 (Empathy), and the unpenalised probes (L1
penalty 0, the default).

**1. Questions** (`pipeline` extra). [`0_dump_questions`](src/0_dump_questions.py) downloads the counselling questions from Hugging Face
and splits them into `train`, `cal` and `test` sets; the probes are trained on answers to `cal`, and steering uses
`test`.

```bash
python -m src.0_dump_questions
```

**2. Traits (optional).** [`1a_prepare_traits`](src/1a_prepare_traits.py) and
[`1b_prepare_instructions`](src/1b_prepare_instructions.py) generate the trait definitions and instruction pairs with
an LLM. Their output is already in `data/traits/`; rerun them only to create new traits.

**3. Activations** (GPU). [`2_prepare_probing_dataset`](src/2_prepare_probing_dataset.py) answers the questions
under the trait's positive and negative instructions and stores the activations of every layer.

```bash
python -m src.2_prepare_probing_dataset trait_id=7 model.checkpoint=apertus_v1.5_8B
```

**4. Probes** (GPU). [`3_train_probes`](src/3_train_probes.py) trains one logistic probe per layer and L1 penalty
(0, 0.01, 0.1, 0.5 and 1 by default) and logs the accuracies to W&B. The probe weights are the trait directions.

```bash
python -m src.3_train_probes trait_id=7 model.checkpoint=apertus_v1.5_8B
```

**5. Geometry.** Once steps 3 and 4 have been run for all 30 traits,
[`4a_analyze_trait_geometry`](src/4a_analyze_trait_geometry.py) compares the directions (similarity, clustering against
the trait categories) and [`4b_plot_trait_geometry`](src/4b_plot_trait_geometry.py) plots the similarity matrix.
[`4d`](src/4d_clustering_scope_check.py) and [`4e`](src/4e_direction_uniqueness_check.py) add checks across models
and penalties.

```bash
python -m src.4a_analyze_trait_geometry --checkpoint apertus_v1.5_8B --l1 0.0
python -m src.4b_plot_trait_geometry --results_json data/plots/probe_geometry/apertus_v1.5_8B/geometry_l1_0.0.json
```

The probe accuracy per layer, [`4c_plot_layer_accuracy`](src/4c_plot_layer_accuracy.py), reads a W&B summary that is
included in the repository, so this figure can be made right after installation:

```bash
python -m src.4c_plot_layer_accuracy --checkpoint apertus_v1.5_8B --l1_values 1.0
```

**6. Steering** (GPU). [`5a_steer_trait`](src/5a_steer_trait.py) answers the held-out questions with the trait's
direction added to the activations at a range of strengths, plus an unsteered baseline.
[`5b_steer_angular`](src/5b_steer_angular.py) instead steers along mixtures of two traits' directions.

```bash
python -m src.5a_steer_trait trait_id=7 model.checkpoint=apertus_v1.5_8B
python -m src.5b_steer_angular trait_a=7 trait_b=11 model.checkpoint=apertus_v1.5_8B
```

**7. Scoring.** [`6a_score_coherence`](src/6a_score_coherence.py) flags degenerate text (CPU), and
[`6b_evaluate_steering`](src/6b_evaluate_steering.py) lets an LLM judge compare each steered response with the
unsteered one (GPU). By default the judge scores the steered trait itself; for cross-trait effects, `eval_trait_id`
takes a list of traits to judge for and `eval_sign=neg` judges for their opposites (see
[`configs/evaluate_steering.yaml`](configs/evaluate_steering.yaml)).

```bash
python -m src.6a_score_coherence trait_id=7 checkpoint=apertus_v1.5_8B
python -m src.6b_evaluate_steering trait_id=7 checkpoint=apertus_v1.5_8B
```

**8. Analyses and figures.** The cross-trait analyses need every trait steered and judged for all 30 traits and their
opposites at strength 0.02. [`7a_build_cross_trait_matrix`](src/7a_build_cross_trait_matrix.py) collects these
judgements into a matrix, [`7c_analyze_cross_trait_correlation`](src/7c_analyze_cross_trait_correlation.py) relates
them to the direction similarities, and [`8b_plot_cross_trait_matrix`](src/8b_plot_cross_trait_matrix.py) plots the
matrix.

```bash
python -m src.7a_build_cross_trait_matrix --checkpoint apertus_v1.5_8B
python -m src.7c_analyze_cross_trait_correlation --checkpoint apertus_v1.5_8B
python -m src.8b_plot_cross_trait_matrix --checkpoint apertus_v1.5_8B
```

The remaining scripts follow the same pattern: [`7b`](src/7b_build_angular_steering.py),
[`7f`](src/7f_angular_summary.py), [`8d`](src/8d_plot_angular_separability.py) and
[`8e`](src/8e_plot_angular_polar.py) analyse the two-trait steering of step 6;
[`7d`](src/7d_semantic_confound_check.py) controls for the textual similarity of the trait descriptions;
[`7e`](src/7e_steering_l1_comparison.py) compares steering with probes of different penalties;
[`8a`](src/8a_plot_steering_eval.py) and [`8c`](src/8c_plot_geometry_vs_transfer.py) make the remaining thesis
figures.

## Running on CSCS

The GPU stages were run on CSCS Alps (Clariden) with the scripts in [`remote/`](remote/). They submit Slurm jobs that
run inside the Swiss AI post-training container, the reference environment of the thesis. On other clusters the
launch and container commands (`srun` with the Pyxis/enroot plugin) may need adapting. The scripts need bash 4.4 or
newer.

Configure once in your checkout on the cluster:

```bash
cp remote/cluster.env.example remote/cluster.env
```

[`cluster.env.example`](remote/cluster.env.example) explains each setting: Slurm account, container image, W&B
project, token files and extra mounts. Its account and image values are the CSCS setup of the thesis and require
access to that allocation and storage. `remote/cluster.env` is not tracked by Git.

Any stage of the pipeline above is submitted as one job by passing the same module and arguments to
[`submit_pipeline.sh`](remote/submit_pipeline.sh), which also sets the job's resources:

```bash
bash remote/submit_pipeline.sh -m src.3_train_probes trait_id=7 model.checkpoint=apertus_v1.5_8B
```

Job logs are written to `remote/logs/`. For many traits at once, [`submit_chain.sh`](remote/submit_chain.sh) and
[`submit_stage.sh`](remote/submit_stage.sh) submit dependent jobs per trait, and
[`reserve_resources.sh`](remote/reserve_resources.sh) starts an interactive node; their usage is at the top of each
script.

The container is built from [swiss-ai/posttraining](https://github.com/swiss-ai/posttraining) at commit `38c6036`, and
[`remote/setup.sh`](remote/setup.sh) and [`remote/edf.toml`](remote/edf.toml) are adapted from that repository. The
submission wrappers were written for this project, following the conventions of the upstream scripts.

## Citation

```bibtex
@misc{pletscher2026activediscovery,
  title  = {Active Discovery of Persona Representations in Large Language Models},
  author = {Wanja Pletscher},
  year   = {2026},
  note   = {Bachelor's thesis, Interactive Visualization \& Intelligence Augmentation Lab, ETH Zurich}
}
```

## Acknowledgements

This work was carried out as a bachelor's thesis at the Interactive Visualization & Intelligence Augmentation (IVIA)
lab, ETH Zürich, supervised by Anna Hedström. I thank Prof. Dr. Mennatallah El-Assady and the IVIA lab for the
opportunity to carry out this project, and the Swiss National Supercomputing Centre (CSCS) for access to the Clariden
computing resources through the Swiss AI Initiative.

## License

The license of this project will be added. The files adapted from swiss-ai/posttraining remain under its MIT License
([`remote/LICENSE-posttraining`](remote/LICENSE-posttraining)).
