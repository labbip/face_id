"""Sentence embeddings of the 150 Jester jokes (data/raw/jester/jokes_texts.xlsx, row i = joke i+1) with a public sentence-transformer.
Writes results/jester_embed_text.npy (150 x d) in joke-id order. Run: uv run --with sentence-transformers python -I scripts/66_jester_embed.py
"""
import numpy as np, pandas as pd
from sentence_transformers import SentenceTransformer
t = pd.read_excel("data/raw/jester/jokes_texts.xlsx", header=None).iloc[:, 0].astype(str).tolist(); print("jokes", len(t))
m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2"); E = m.encode(t, normalize_embeddings=True, batch_size=32); np.save("results/jester_embed_text.npy", E.astype(np.float32)); print(E.shape)
