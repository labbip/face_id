"""Compact table of the XPASS follow-ups (77): n_est=100 seen/unseen r for D0/D1/D2 and the paired D1-D0, D2-D1 deltas, from results/<src>_{summary,paired}.csv.
Run: uv run python -I scripts/78_xpass_followup_summary.py  -> results/xpass_followup_table.csv
"""
import os, pandas as pd
SRC = [("LAPIS full (553 raters)", "lapis_img"), ("LAPIS 129 raters, all ratings", "lapis_r129_img"), ("LAPIS 129 raters x 227 ratings", "lapis_thin129_img"),
       ("XPASS art (own models)", "xpass_art_img"), ("XPASS art raters <- LAPIS models", "lapis_on_xpass"), ("LAPIS raters on XPASS images <- XPASS models", "xpass_on_lapis"),
       ("XPASS fashion", "xpass_fashion_img"), ("XPASS scenery", "xpass_scenery_img"),
       ("XPASS art, Like", "xpass_art_Like_img"), ("XPASS fashion, Like", "xpass_fashion_Like_img"), ("XPASS scenery, Like", "xpass_scenery_Like_img"),
       ("XPASS art, Beautiful", "xpass_art_Beautiful_img"), ("XPASS fashion, Beautiful", "xpass_fashion_Beautiful_img"), ("XPASS scenery, Beautiful", "xpass_scenery_Beautiful_img")]
rows = []
for name, src in SRC:
    if not os.path.exists(f"results/{src}_summary.csv"): continue
    S = pd.read_csv(f"results/{src}_summary.csv"); P = pd.read_csv(f"results/{src}_paired.csv")
    for n in sorted(S.n_est.unique()):
        s = S[S.n_est == n].set_index("kind"); p = P[P.n_est == n]
        def dl(c, side): q = p[(p.contrast == c) & (p.side == side)]; return (q.delta.iloc[0], q.p_t.iloc[0], int(q.n.iloc[0])) if len(q) else (float("nan"),) * 3
        r = dict(data=name, src=src, n_est=n, n_raters=dl("D1 - D0", "seen")[2], pop_seen=s.loc["D0", "r_pop_seen"], D0_seen=s.loc["D0", "r_amort_seen"], D1_seen=s.loc["D1", "r_amort_seen"], D2_seen=s.loc["D2", "r_amort_seen"], D0_unseen=s.loc["D0", "r_amort_unseen"], D1_unseen=s.loc["D1", "r_amort_unseen"], D2_unseen=s.loc["D2", "r_amort_unseen"])
        for c, k in (("D1 - D0", "d10"), ("D2 - D1", "d21")):
            for side in ("seen", "unseen"): d, pv, _ = dl(c, side); r[f"{k}_{side}"] = d; r[f"{k}_{side}_p"] = pv
        rows.append(r)
T = pd.DataFrame(rows); T.to_csv("results/xpass_followup_table.csv", index=False)
pd.set_option("display.width", 250); print(T[T.n_est == 100].drop(columns=["src", "n_est"]).round(3).to_string(index=False)); print("\nn_est=50:"); print(T[T.n_est == 50][["data", "D0_seen", "D1_seen", "D2_seen", "d10_seen", "d21_seen", "d10_unseen", "d21_unseen"]].round(3).to_string(index=False))
