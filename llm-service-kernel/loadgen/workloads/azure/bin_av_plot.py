import pandas as pd
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

csv_path = "/home/mohamad/LLM-end-to-end-Service-main/llm-service-kernel/loadgen/workloads/azure/AzureLLMInferenceTrace_conv.csv"    # change
out_path = "gen_vs_ctx_binned.png"
bins = 30

df = pd.read_csv(csv_path)
col_map = {c.lower().replace(" ", ""): c for c in df.columns}
ctx_col = col_map.get("contexttokens")
gen_col = col_map.get("generatedtokens")

ctx = pd.to_numeric(df[ctx_col], errors="coerce")
gen = pd.to_numeric(df[gen_col], errors="coerce")
m = ctx.notna() & gen.notna()
ctx = ctx[m].astype(int)
gen = gen[m].astype(int)

# Bin ctx into equal-width bins
edges = np.linspace(ctx.min(), ctx.max(), bins + 1)
bin_id = np.digitize(ctx, edges) - 1
bin_id = np.clip(bin_id, 0, bins - 1)

# Compute median gen per bin (more robust than mean)
med = np.array([np.median(gen[bin_id == i]) if np.any(bin_id == i) else np.nan for i in range(bins)])
centers = (edges[:-1] + edges[1:]) / 2

plt.figure()
plt.plot(centers, med, marker="o", linestyle="-")
plt.xlabel("ContextTokens (bin center)")
plt.ylabel("Median GeneratedTokens")
plt.title("Median GeneratedTokens vs ContextTokens (binned)")
plt.tight_layout()
plt.savefig(out_path, dpi=200)
print(f"Wrote {out_path}")
plt.show()
