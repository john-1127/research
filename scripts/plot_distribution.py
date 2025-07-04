import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("./analysis/Pearson_detailed.csv")
df = df[df['Group'].astype(int) <= 3]
df['Group'] = df['Group'].astype(int).apply(lambda x: f'Group{x}')
plt.figure(figsize=(10, 5))

palette = {
    'FFNN': '#A0A2A3',
    'HQNN': '#009495',
    'Ensemble':'#CC252B'
}

palette = {
    'FFNN': '#0072B2',
    'HQNN': '#D55E00',   
    'Ensemble': '#009E73'
}

sns.boxplot(data=df, x='Group', y='Pearson', hue='Model', showfliers=False, width=0.6, whis=0)

legend = plt.legend(title=None, frameon=False, loc='lower right')
plt.title("Pearson Correlation Coefficient Distribution of Different Models Across Groups")
plt.ylabel("Pearson Correlation Coefficient")
plt.xlabel("")
plt.ylim(0.85, 0.97)
plt.tight_layout()
plt.savefig("./output/figures/Pearson_1-3.png", dpi=300)
