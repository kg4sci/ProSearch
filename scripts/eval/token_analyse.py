import json, os, math
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast

domain = [
    "cell_biology_culture",
    "biochemical",
    "immunological",
    "electrochemistry",
    "organic_chemistry",
    "nanoparticle",
    "solid_state_inorganic",
    "solution_based_inorganic",
]


_gpt2_tok = GPT2TokenizerFast.from_pretrained("model/gpt2")

def gpt2_token_count(text: str) -> int:
    return len(_gpt2_tok.encode(text))

lengths_by_domain = {}
stats_by_domain = {}

# ====== 统计 token 长度 ======
for d in domain:
    path = f"seed_data/{d}/paragraphs.json"
    if not os.path.exists(path):
        print(f"[WARN] File not found, skipping: {path}")
        continue

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    lens = [gpt2_token_count(rec["paragraph"])
            for rec in data
            if isinstance(rec, dict) and "paragraph" in rec and rec["paragraph"]]

    lengths_by_domain[d] = lens
    if lens:
        stats_by_domain[d] = {
            "count": len(lens),
            "mean": sum(lens) / len(lens),
            "median": sorted(lens)[len(lens)//2],
            "min": min(lens),
            "max": max(lens),
        }
        print(f"Domain: {d:25s} | Count: {len(lens):6d} | "
              f"Mean: {stats_by_domain[d]['mean']:.2f} | "
              f"Median: {stats_by_domain[d]['median']} | "
              f"Min/Max: {stats_by_domain[d]['min']}/{stats_by_domain[d]['max']}")
    else:
        print(f"Domain: {d} has no paragraphs.")

# ====== 画多子图直方图 ======
n = len(lengths_by_domain)
cols = 4 if n > 1 else 1
rows = math.ceil(n / cols)

plt.figure(figsize=(12, 9))

for i, (d, lens) in enumerate(lengths_by_domain.items(), start=1):
    ax = plt.subplot(rows, cols, i)

    if not lens:
        ax.set_title(d + " (empty)")
        ax.axis("off")
        continue

    ax.hist(lens, bins=30)
    ax.set_title(d)
    ax.set_xlabel("GPT-2 token length")
    ax.set_ylabel("Count")

    # mean_val = stats_by_domain[d]["mean"]
    # ax.axvline(mean_val, linestyle="--")
    # ax.text(mean_val, ax.get_ylim()[1]*0.9, f"mean={mean_val:.1f}",
    #         rotation=90, va="top")

plt.tight_layout()

out_path = "data/paragraph_length_distributions.png"
plt.savefig(out_path, dpi=200)
print(f"\nSaved figure to: {out_path}")
plt.show()



# cd /home/liangwenliang/检索数据集
# conda activate retrieve_dataset
# python scripts/eval/token_analyse.py

