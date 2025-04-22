import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./output/figures/distribution.csv")

plt.figure(figsize=(10, 5))

palette = {
    'FFNN': '#A0A2A3',
    'HQNN': '#009495',
    'Ensemble': '#803546'
}

sns.boxplot(data=df, x='Group', y='SIS', hue='Model', showfliers=False, palette=palette, width=0.6, whis=0)

legend = plt.legend(title=None, frameon=False, loc='lower right')
plt.title("SIS Score Distribution of Different Models Across Groups")
plt.ylabel("Spectral Information Similarity (SIS)")
plt.xlabel("Groups")
plt.ylim(0.84, 0.96)
plt.tight_layout()
plt.savefig("sis_boxplot.png", dpi=300)
