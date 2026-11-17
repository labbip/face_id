# Source datasets and what this repository contains from them

This repository contains **aggregate values only** (means over raters, test statistics, histogram densities, the plotted values of the
figures), with the one exception named in the London Set row. It contains no stimulus, no rating or judgment of an individual participant, no other per-participant model output, no latent
representation, no stimulus embedding and no trained model (trained models hold stimulus embeddings learned from the ratings).
The datasets must be obtained from their original sources under their own terms.

| Dataset | Terms stated by the source | In this repository |
|---|---|---|
| Xu, Keles, Lin & Adolphs 2026 (OSF nm7y8) | no licence stated on the OSF project | aggregate values |
| London Set (DeBruine & Jones) | CC BY | aggregate values; loadings of the 102 faces on a principal component (`results/figdata/london_resid_pc1.csv`); **per-rater exception**: `results/figdata/london_z_prefmale.csv`, the 2,501 points of the panel `london_z_prefmale.pdf` (for each rater the male-face preference index, i.e. mean residual rating of male minus female faces, and its prediction from the latent variable; no identifiers, no ratings) |
| AVFS (Andrews, Joniak & Xiang 2023) | CC BY-NC-SA 4.0 | aggregate values |
| FaceSim3D (Hofmann et al.) | CC BY | aggregate values |
| SCUT-FBP5500 (Liang et al. 2018) | research use only | aggregate values |
| LAPIS (Maerten et al. 2025) | CC BY-NC 4.0 | aggregate values |
| AADB (Kong et al. 2016) | research use | aggregate values |
| EVA (Kang, Valenzise & Dufaux 2020) | CC0 | aggregate values |
| Jester (Goldberg et al. 2001) | research use with citation | aggregate values |
| Keller & Vosshall 2016 | article's additional file (open access) | aggregate values |
| XPASS-Vis (Hayashi et al. 2026) | password-protected archive, non-commercial research | nothing (no result file, statistic or log; the results are reported in the paper) |

The aggregate results, figure source data and figures of this repository are released under CC BY 4.0; the code under the MIT licence.
