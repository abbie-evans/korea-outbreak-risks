import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Figure 1C

data = {
    'year': [2000, 2025, 2050],
    '0-4': [3259783, 1219810, 1152490],
    '4-9': [3521464, 1766989, 1246698],
    '10-14': [3129982, 2271667, 1352080],
    '15-19': [3842432, 2267007, 1357147],
    '20-24': [3854382, 2660295, 1325750],
    '25-29': [4352913, 3476469, 1471574],
    '30-34': [4247992, 3670087, 2039223],
    '35-39': [4273079, 3305481, 2529495],
    '40-44': [4020438, 3851246, 2503915],
    '45-49': [2921443, 3865770, 2793071],
    '50-54': [2365862, 4367799, 3504568],
    '55-59': [2006389, 4292571, 3653864],
    '60-64': [1817056, 4155466, 3269232],
    '65-69': [1381212, 3688605, 3742776],
    '70-74': [922213, 2527085, 3632046],
    '75+': [608084+483387, 1820534+2477683, 3871945+7661086],
}

df = pd.DataFrame(data)
# Set year as index
df.set_index('year', inplace=True)

# Convert counts to proportions (row-wise)
df_prop = df.div(df.sum(axis=1), axis=0)

colors = [plt.cm.Blues(i) for i in np.linspace(0.2, 1, 4)] + \
         [plt.cm.Reds(i) for i in np.linspace(0.2, 1, 6)] + \
         [plt.cm.Greens(i) for i in np.linspace(0.2, 1, 6)]

df_prop.plot(
    kind='bar',
    stacked=True,
    color=colors,
    figsize=(6, 6),
    width=0.8
)

fig = plt.gcf()
fig.set_size_inches(8, 6)
plt.xticks(rotation=0, fontsize=18)
plt.yticks(fontsize=18)
plt.ylabel('Proportion of population', labelpad=10, fontsize=20)
plt.xlabel('Year', labelpad=10, fontsize=20)
plt.legend(title='Age group', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=14, title_fontsize=14)
plt.ylim(0, 1)
# add total population on top of each bar
totals = df.sum(axis=1)
for i, total in enumerate(totals):
    plt.text(i, 1.02, f'{total:,}', ha='center', va='bottom', fontsize=14)
plt.tight_layout()
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.savefig('Figures/Proportions_korea.svg', bbox_inches='tight')
plt.show()

# Figure 1B

C = pd.read_csv('Data/contact_matrix_2020.csv', header=None, dtype=float).values

plt.figure(figsize=(8, 6))
sns.heatmap(C.T, yticklabels=range(1, 16), cmap='Blues', vmin=0, vmax=2)
plt.gca().invert_yaxis()
colorbar = plt.gcf().axes[-1]
colorbar.set_ylabel('Daily contacts', labelpad=10, fontsize=20)
colorbar.tick_params(labelsize=18)
# have top of colorbar values say 2+
colorbar.set_yticks([0, 0.5, 1, 1.5, 2])
colorbar.set_yticklabels(['0', '0.5', '1', '1.5', '2+'])
plt.xlabel('Age of individual', labelpad=10, fontsize=24)
plt.ylabel('Age of contact', labelpad=10, fontsize=24)
plt.xticks(np.arange(0, 16), labels=['0', '', '10', '', '20', '', '30', '', '40', '', '50', '', '60', '', '70', ''], fontsize=20)
plt.yticks(np.arange(0, 16), labels=['0', '', '10', '', '20', '', '30', '', '40', '', '50', '', '60', '', '70', ''], fontsize=20, rotation=0)
plt.tight_layout()
plt.savefig(f'Figures/Contact_matrix_2020.svg', bbox_inches='tight')
plt.show()

total_contacts = np.sum(C, axis=1)
plt.figure(figsize=(8, 4))
plt.bar(range(1, 17), total_contacts, width=1, color='lightblue', edgecolor='black', linewidth=1)
plt.ylabel('Total daily contacts', labelpad=10, fontsize=24)
plt.xticks([])
plt.yticks(fontsize=20)
plt.xlim([0.5, 16.75])
plt.ylim(top=35)
ax = plt.gca()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'Figures/Total_contacts_2020.svg', bbox_inches='tight')
plt.show()
