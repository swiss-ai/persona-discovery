"""Control the geometry-behaviour correlation for the textual similarity of the trait definitions.

Traits with similar meaning may produce similar probing prompts, hence similar probe directions, and
answers that a judge scores similarly. This script measures how similar the trait texts are and
partials that similarity out of the correlation between probe-direction cosine and cross-trait
transfer (src/7c_analyze_cross_trait_correlation.py).

Text similarity is measured in two ways: a TF-IDF cosine (shared wording) and two sentence-embedding
models (shared meaning), all-MiniLM-L6-v2 (primary) and all-mpnet-base-v2; results for both models
are reported. Each is computed on three text variants: `positive` (positive name and description),
`contrast` (positive and negative name and description) and `full` (`contrast` plus all instruction
pairs).

Statistics as in src/7c_analyze_cross_trait_correlation.py: cells sharing a trait are not
independent, so matrix-level p-values are Mantel tests over joint row/column label permutations, and
the partial correlation is reported alongside a Mantel null built the same way.

Reads data/plots/probe_geometry/<checkpoint>/cross_trait_matrix.json and data/traits/, downloads the
two embedding models from Hugging Face, and writes semantic_confound_check.json next to the input.

    python -m src.7d_semantic_confound_check
"""
import os, json, re, datetime
import numpy as np
import fire
import torch
from scipy.stats import spearmanr, rankdata
from transformers import AutoTokenizer, AutoModel

from src.utils.paths import repo_path

# Each model is used at its own documented max_seq_length (from its sentence_bert_config.json).
MODELS = {
    "all-MiniLM-L6-v2": ("sentence-transformers/all-MiniLM-L6-v2", 256),
    "all-mpnet-base-v2": ("sentence-transformers/all-mpnet-base-v2", 384),
}


def _embed(texts, repo, max_len):
    """Mean-pool over non-padding tokens, then L2-normalise.

    This is the recipe both models declare in their own 1_Pooling/config.json
    (pooling_mode_mean_tokens: true, no CLS, no max), so it reproduces sentence-transformers.
    Returns the embeddings, the number of inputs longer than `max_len` (they are truncated, which
    is reported with the results) and the longest input length in tokens.
    """
    tok = AutoTokenizer.from_pretrained(repo)
    mod = AutoModel.from_pretrained(repo).eval()
    lens = [len(tok(t)["input_ids"]) for t in texts]
    n_trunc = sum(1 for L in lens if L > max_len)
    enc = tok(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
    with torch.no_grad():
        out = mod(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    emb = (out * m).sum(1) / m.sum(1).clamp(min=1e-9)
    return torch.nn.functional.normalize(emb, dim=1).numpy(), n_trunc, int(max(lens))


def _tfidf_cosine(docs, stopwords):
    """Cosine similarity between trait texts under a TF-IDF weighting.

    Included alongside the sentence embeddings because it captures a different confound: literal
    shared vocabulary rather than shared meaning. Two traits can be conceptually close while sharing
    almost no words, and conversely. Smoothed inverse document frequency, as in the common
    formulation, with L2-normalised rows so the dot product is a cosine.
    """
    toks = [[w for w in re.findall(r"[a-z]+", d.lower()) if w not in stopwords and len(w) > 2] for d in docs]
    vocab = sorted({w for t in toks for w in t})
    idx = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(len(vocab))
    for t in toks:
        for w in set(t):
            df[idx[w]] += 1
    idf = np.log((1 + len(toks)) / (1 + df)) + 1
    M = np.zeros((len(toks), len(vocab)))
    for r, t in enumerate(toks):
        for w in t:
            M[r, idx[w]] += 1
        if M[r].sum():
            M[r] /= M[r].sum()
    M *= idf
    M /= np.linalg.norm(M, axis=1, keepdims=True)
    return M @ M.T


def _partial(x, y, z):
    """Spearman partial correlation of x and y given z, by rank residualisation."""
    rx, ry, rz = (rankdata(v).astype(float) for v in (x, y, z))
    Z = np.c_[rz, np.ones_like(rz)]
    res = lambda a: a - Z @ np.linalg.lstsq(Z, a, rcond=None)[0]
    return float(np.corrcoef(res(rx), res(ry))[0, 1])


def _mantel(A, B, off, n, n_perm, seed=0):
    rng = np.random.default_rng(seed)
    obs = spearmanr(A[off], B[off]).correlation
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(spearmanr(A[off], B[np.ix_(p, p)][off]).correlation) >= abs(obs) - 1e-12:
            ge += 1
    return float(obs), (ge + 1) / (n_perm + 1)


def _mantel_partial(C, Bh, S, off, n, n_perm, seed=0):
    """Null for the partial correlation: permute the behavioural matrix only, so the observed
    cosine-semantic dependence is preserved under the null and only the behaviour link is broken."""
    rng = np.random.default_rng(seed)
    obs = _partial(C[off], Bh[off], S[off])
    ge = 0
    for _ in range(n_perm):
        p = rng.permutation(n)
        if abs(_partial(C[off], Bh[np.ix_(p, p)][off], S[off])) >= abs(obs) - 1e-12:
            ge += 1
    return float(obs), (ge + 1) / (n_perm + 1)


def main(checkpoint: str = "apertus_v1.5_8B", n_perm: int = 5000, seed: int = 0, out_dir: str = None):
    out_dir = out_dir or repo_path("data", "plots", "probe_geometry", checkpoint)
    src = json.load(open(os.path.join(out_dir, "cross_trait_matrix.json")))
    ids = src["trait_ids"]
    R = np.array(src["rate"], float)
    D = np.array(src["d"], float)
    C = np.array(src["cosine_similarity"], float)
    n = len(ids)
    off = ~np.eye(n, dtype=bool)

    traits = {t["trait_id"]: t for t in json.load(open(repo_path("data", "traits", "all_traits.json")))["therapist_traits"]}
    instr = {}
    for f in os.listdir(repo_path("data", "traits")):
        m = re.match(r"^(\d+)_", f)
        if m and f.endswith(".json"):
            j = json.load(open(repo_path("data", "traits", f)))
            if "instruction" in j:
                instr[int(m.group(1))] = j["instruction"]

    # Three text variants: "positive" is the trait as defined; "contrast" adds the negative pole,
    # i.e. the contrast the probes separate; "full" adds the instruction pairs used for probing.
    variants = {
        "positive": lambda t: f"{traits[t]['trait_name_pos']}. {traits[t]['base_description_pos']}",
        "contrast": lambda t: (f"{traits[t]['trait_name_pos']}. {traits[t]['base_description_pos']} "
                               f"Versus {traits[t]['trait_name_neg']}. {traits[t]['base_description_neg']}"),
        "full": lambda t: (f"{traits[t]['trait_name_pos']}. {traits[t]['base_description_pos']} "
                           f"Versus {traits[t]['trait_name_neg']}. {traits[t]['base_description_neg']} "
                           + " ".join(x.get("pos", "") + " " + x.get("neg", "") for x in instr.get(t, []))),
    }

    base_rate, p_rate = _mantel(C, R, off, n, n_perm, seed)
    base_d, p_d = _mantel(C, D, off, n, n_perm, seed)
    print(f"[INFO] {checkpoint}: baseline rho(cosine, rate) = {base_rate:+.3f} (Mantel p={p_rate:.5f})")
    print(f"[INFO] {checkpoint}: baseline rho(cosine, d)    = {base_d:+.3f} (Mantel p={p_d:.5f})\n")

    STOP = set("the a an of to and or in on for with that this it is are be as by their his her its them they "
               "you your therapist client clients responses response should while without any not no can may".split())

    results = {}
    print("=" * 78); print("LEXICAL REPRESENTATION: TF-IDF cosine over the same texts")
    results["tfidf"] = {}
    for vname, fn in variants.items():
        T = _tfidf_cosine([fn(t) for t in ids], STOP)
        r_tc, p_tc = _mantel(T, C, off, n, n_perm, seed)
        r_tr, p_tr = _mantel(T, R, off, n, n_perm, seed)
        pr, ppr = _mantel_partial(C, R, T, off, n, n_perm, seed)
        pd_, ppd = _mantel_partial(C, D, T, off, n, n_perm, seed)
        results["tfidf"][vname] = {
            "similarity_offdiag_mean": float(T[off].mean()),
            "rho_similarity_vs_cosine": r_tc, "p_similarity_vs_cosine": p_tc,
            "rho_similarity_vs_rate": r_tr, "p_similarity_vs_rate": p_tr,
            "partial_rho_cosine_rate_given_similarity": pr, "p_partial_rate": ppr,
            "partial_rho_cosine_d_given_similarity": pd_, "p_partial_d": ppd,
            "retained_fraction_rate": pr / base_rate if base_rate else None,
        }
        print(f"  text={vname:<9} sim mean={T[off].mean():+.3f}  sim~cos={r_tc:+.3f}  sim~rate={r_tr:+.3f}(p={p_tr:.4f})")
        print(f"  {'':<14} PARTIAL cos~rate|sim = {pr:+.3f} (retains {100*pr/base_rate:.0f}%)   "
              f"cos~d|sim = {pd_:+.3f}")

    for mname, (repo, max_len) in MODELS.items():
        print("=" * 78); print(f"MODEL: {mname}  (native max_seq_length = {max_len})")
        results[mname] = {}
        for vname, fn in variants.items():
            texts = [fn(t) for t in ids]
            E, n_trunc, max_tok = _embed(texts, repo, max_len)
            S = E @ E.T
            r_cs, p_cs = _mantel(S, C, off, n, n_perm, seed)
            r_sr, p_sr = _mantel(S, R, off, n, n_perm, seed)
            r_sd, p_sd = _mantel(S, D, off, n, n_perm, seed)
            pr, ppr = _mantel_partial(C, R, S, off, n, n_perm, seed)
            pd_, ppd = _mantel_partial(C, D, S, off, n, n_perm, seed)
            results[mname][vname] = {
                "semantic_sim_offdiag_mean": float(S[off].mean()),
                "semantic_sim_offdiag_min": float(S[off].min()),
                "semantic_sim_offdiag_max": float(S[off].max()),
                "rho_semantic_vs_cosine": r_cs, "p_semantic_vs_cosine": p_cs,
                "rho_semantic_vs_rate": r_sr, "p_semantic_vs_rate": p_sr,
                "rho_semantic_vs_d": r_sd, "p_semantic_vs_d": p_sd,
                "partial_rho_cosine_rate_given_semantic": pr, "p_partial_rate": ppr,
                "partial_rho_cosine_d_given_semantic": pd_, "p_partial_d": ppd,
                "retained_fraction_rate": pr / base_rate if base_rate else None,
                "retained_fraction_d": pd_ / base_d if base_d else None,
                "max_seq_length": max_len,
                "longest_input_tokens": max_tok,
                "n_texts_truncated": n_trunc,
            }
            trunc_flag = f"  ⚠ TRUNCATED {n_trunc}/30 (longest {max_tok} tok)" if n_trunc else f"  (longest {max_tok} tok, none truncated)"
            print(f"  text={vname:<9} semsim mean={S[off].mean():+.3f}  "
                  f"sem~cos={r_cs:+.3f}(p={p_cs:.4f})  sem~rate={r_sr:+.3f}(p={p_sr:.4f})  "
                  f"sem~d={r_sd:+.3f}(p={p_sd:.4f}){trunc_flag}")
            print(f"  {'':<14} PARTIAL cos~rate|sem = {pr:+.3f} (p={ppr:.5f}, retains {100*pr/base_rate:.0f}%)   "
                  f"cos~d|sem = {pd_:+.3f} (p={ppd:.5f}, retains {100*pd_/base_d:.0f}%)")

    payload = {
        "_meta": {
            "description": "Semantic-similarity control for the geometry-behaviour correlation "
                           "(cross_trait_correlation.json).",
            "question": "Is rho(probe cosine, behavioural transfer) explained by semantic similarity "
                        "of the trait definitions?",
            "models": {k: {"repo": v[0], "max_seq_length": v[1]} for k, v in MODELS.items()},
            "lexical_control": "A TF-IDF cosine over the same three text variants is computed alongside the "
                               "embedding models. It captures shared vocabulary rather than shared meaning, and "
                               "removes more of the correlation than the semantic control does. Smoothed idf, "
                               "L2-normalised rows.",
            "model_choice": "all-MiniLM-L6-v2 is the primary model (chosen in advance); all-mpnet-base-v2 "
                            "is a robustness check. Both are reported.",
            "embedding": "mean-pool over non-padding tokens of the last hidden state, then L2 "
                         "normalise; similarity is the dot product of normalised vectors (cosine).",
            "text_variants": {
                "positive": "positive trait name + positive description (the trait as defined)",
                "contrast": "positive and negative name + description (the contrast the probe separates)",
                "full": "contrast plus all three instruction pairs actually used as prompts",
            },
            "statistics": f"Mantel tests, {n_perm} joint row/column label permutations. The partial "
                          "correlation's null permutes the BEHAVIOURAL matrix only, so the observed "
                          "cosine-semantic dependence is preserved under the null.",
            "truncation": "Each model is run at its OWN documented max_seq_length (MiniLM 256, mpnet "
                          "384). Any variant still exceeding that is flagged per result via "
                          "n_texts_truncated. Truncation weakens the semantic covariate and would "
                          "therefore bias the partial correlation upward (in favour of the main "
                          "result), so a truncated variant is not the strongest control.",
            "limitations": "Sentence embeddings are themselves a model's notion of similarity, not "
                           "ground truth. A low semantic-behaviour correlation does not prove the "
                           "traits are semantically unrelated, only that this embedding does not "
                           "capture a relation that predicts transfer. n=30 traits bounds the power "
                           "of every matrix-level test here.",
            "generated": datetime.datetime.now().isoformat(timespec="seconds"),
        },
        "baseline": {"rho_cosine_rate": base_rate, "p_cosine_rate": p_rate,
                     "rho_cosine_d": base_d, "p_cosine_d": p_d},
        "results": results,
    }
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "semantic_confound_check.json")
    json.dump(payload, open(out_path, "w"), indent=2)
    print(f"\n[INFO] wrote {out_path}")


if __name__ == "__main__":
    fire.Fire(main)
