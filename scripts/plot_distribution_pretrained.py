import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


df = pd.read_csv("./output/sis/pretrained_three_model_comparison.csv")

df = df[df['Model'].isin(['FFNN', 'HQNN', 'Ensemble'])]

plt.figure(figsize=(8, 5))

# palette = {
#     'FFNN': '#0072B2',
#     'HQNN': '#D55E00',
#     'Ensemble': '#009E73',
# }

ax = sns.boxplot(data=df, x='Group', y='SIS', hue='Model', showfliers=False, width=0.6, whis=0)

# plt.title("SIS Score Distribution of Direct and Pretrained")


legend = plt.legend(title=None, frameon=False, loc='lower right')
ax.set_xticks([])
plt.ylabel("Spectral Information Similarity (SIS)")
plt.xlabel("")
plt.ylim(0.65, 0.92)
plt.yticks(np.arange(0.65, 0.921, 0.05))
plt.tight_layout()
plt.savefig("./test1234.png", dpi=300)
