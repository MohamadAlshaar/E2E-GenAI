import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "/home/mohamad/LLM-end-to-end-Service-main/llm-service-kernel/loadgen/workloads/azure/AzureLLMInferenceTrace_conv.csv"   # change
out_path = "ctx_gen_heatmap.png"

df = pd.read_csv(csv_path)
col_map = {c.lower().replace(" ", ""): c for c in df.columns}
ctx_col = col_map.get("contexttokens")
gen_col = col_map.get("generatedtokens")
if ctx_col is None or gen_col is None:
    raise SystemExit(f"Need ContextTokens + GeneratedTokens. Columns: {df.columns.tolist()}")

ctx = pd.to_numeric(df[ctx_col], errors="coerce")
gen = pd.to_numeric(df[gen_col], errors="coerce")
m = ctx.notna() & gen.notna()
ctx = ctx[m].astype(int)
gen = gen[m].astype(int)

plt.figure()
plt.hist2d(ctx, gen, bins=60)   # increase bins for more detail
plt.xlabel("ContextTokens")
plt.ylabel("GeneratedTokens")
plt.title("2D frequency of (ContextTokens, GeneratedTokens)")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"Wrote {out_path} (n={len(ctx)})")
plt.show()
