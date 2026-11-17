"""Parsers for Meadows multi-arrangement data (Xu et al. 2026, OSF nm7y8).

Output: one long-format table of placements with columns
    exp, source, pid, task, trial, face, x, y, excluded
where ``task`` is 'MA' for the main multi-arrangement task and
'TR_set{k}_rep{r}' for test-retest trials; ``excluded`` follows the paper's
trial-exclusion rules (mean inter-placement time < 0.9 s, or multi-select
without readjustment of at least half of the multi-selected faces).
"""
from __future__ import annotations
import json
import numpy as np
import pandas as pd

PLACEMENT_T = 0.9  # seconds


def _log_df(log_list):
    return pd.DataFrame(log_list, columns=["time", "event", "note"]).sort_values("time")


def trial_exclusion_flag(log_list) -> tuple[bool, dict]:
    """Return (exclude, info) replicating utility_function rules."""
    df = _log_df(log_list)
    placed = df[df.event == "placed"]
    info = {"n_placed": len(placed), "mean_dt": np.nan, "multi": False, "readjust_ok": True}
    if len(placed) == 0:
        return True, info
    times, counts = np.unique(placed.time.to_numpy(), return_counts=True)
    if len(times) > 1:
        dt = np.ediff1d(times)
    else:
        start = df.loc[df.event == "logStarted", "time"].to_numpy()
        dt = np.array([times[0] - (start[0] if len(start) else times[0])])
    info["mean_dt"] = float(np.mean(dt))
    exclude = info["mean_dt"] < PLACEMENT_T * 1000
    if np.any(counts > 1):
        info["multi"] = True
        ids = placed.note.apply(lambda d: d["id"] if isinstance(d, dict) else None).to_numpy()
        t_arr = placed.time.to_numpy()
        n_multi = 0; n_re = 0
        for t in times[counts > 1]:
            rep = ids[t_arr == t]
            n_multi += len(rep)
            later = ids[t_arr > t]
            n_re += sum(1 for i in rep if np.any(later == i))
        info["readjust_ok"] = n_re >= 0.5 * n_multi
        if not info["readjust_ok"]:
            exclude = True
    return bool(exclude), info


def separate_log_into_trials(log_list):
    """Split one task-level log (Exp2 tree.json) into per-trial logs."""
    df = pd.DataFrame(log_list, columns=["time", "event", "note"])
    starts = df.index[df.event == "logStarted"].to_list()
    out = []
    for i, s in enumerate(starts):
        e = starts[i + 1] if i + 1 < len(starts) else len(df)
        out.append(log_list[s:e])
    return out


def parse_exp1_tree(path: str, pid: str, ma_task_index: int = 7):
    """Parse one Exp1 stage-2 tree.json → (placements DataFrame, meta dict)."""
    with open(path) as fo:
        d = json.load(fo)
    task = d["tasks"][ma_task_index]
    rows = []
    excl = []
    for t, trial in enumerate(task["trials"]):
        ex, info = trial_exclusion_flag(trial["log"]) if "log" in trial else (False, {})
        excl.append(ex)
        for p in trial["positions"]:
            rows.append((pid, "MA", t, p["name"], float(p["x"]), float(p["y"]), ex))
    df = pd.DataFrame(rows, columns=["pid", "task", "trial", "face", "x", "y", "excluded"])
    meta = {"pid": pid, "n_trials": len(task["trials"]), "n_excluded": int(sum(excl)),
            "n_stimuli": len(task["stimuli"]), "min_evidence": task.get("min_evidence", np.nan)}
    return df, meta, d


def parse_exp1_questionnaire(d, q_index=8, panas_index=9, bfi_index=10):
    """Flatten the questionnaire / PANAS / BFI tasks of an Exp1 tree dict."""
    skip = {"status", "task", "stimuli", "trials", "Reminder", "qualification"}
    out = {}
    for idx, prefix in [(q_index, ""), (panas_index, "PANAS_"), (bfi_index, "BFI_")]:
        if idx >= len(d["tasks"]):
            continue
        for k, v in d["tasks"][idx].items():
            if k in skip:
                continue
            key = prefix + "_".join(k.split(" ")) if prefix else k
            if prefix and isinstance(v, str) and " = " in v:
                try:
                    v = int(v.split(" = ")[0])
                except ValueError:
                    pass
            out[key] = v
    return out
