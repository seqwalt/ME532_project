import numpy as np
import pandas as pd
import seaborn as sns
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from DataPreprocess import one_hot, normalize, low_rank

# --- Load data --- #
vis_data = pd.read_csv("../data/cardio_disease.csv", sep=';')
A_d = np.genfromtxt('../data/cardio_disease.csv',delimiter=';',skip_header=1)
Araw = A_d[:,0:11]
d = A_d[:,11].reshape(-1,1)
d[d == 0] = -1

# --- Display Correlation Heatmap --- #
plt.figure(figsize=(12, 10))
heatmap = sns.heatmap(vis_data.corr(),
        annot=True, linewidths=0, vmin=-1,
        cmap=sns.diverging_palette(20, 220, n=200)
)
heatmap.set_xticklabels(
    heatmap.get_xticklabels(),
    rotation=40,
    horizontalalignment='right'
);

# --- Data Preprocessing --- #
#Feature scaling
A = normalize(Araw)
# one-hot encoding for gender, cholesterol, glucose, smoke, alcohol and
# physical activity
A = one_hot(A,[1,6,7,8,9,10])

# Plot some singular values
U0,s0,VT0 = LA.svd(Araw,full_matrices=False)
fig = plt.figure(figsize=(12,4), dpi= 100)
ax = fig.add_subplot(121)
ax.plot(s0,'-o') # Plot sing vals for original data

rank = 14
Ar, s2 = low_rank(A, rank)
ax.plot(s2,'^-') # Plot sing vals for low-rank approx of
                           # preprocessed data
ax.set_xlabel('Index $i$', fontsize=17)
ax.set_ylabel('$\sigma_i$', fontsize=17)
ax.set_title('Linear Scale', fontsize=18)
ax.legend(('Original data','Low-rank and normalized'),fontsize=13)
ax.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
ax.yaxis.offsetText.set_fontsize(12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)


# Plot sing vals for preprocessed data
# U1,s1,VT1 = LA.svd(A,full_matrices=False)
# fig = plt.figure(figsize=(6,4), dpi= 100)
# ax.plot(np.log10(s1),'-o')
# ax.set_xlabel('Singular value index $i$', fontsize=16)
# ax.set_title('Singular Values', fontsize=18)

# Plot original and low-rank preprocessed,
# but log of the sing vals.
ax1 = fig.add_subplot(122)
ax1.plot(np.log10(s0),'-o')
ax1.set_xlabel('Index $i$', fontsize=17)
ax1.set_ylabel('$\log(\sigma_i)$', fontsize=17)
ax1.set_title('Logarithmic Scale', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
ax1.plot(np.log10(s2),'^-')
ax1.legend(('Original data','Low-rank and normalized'),fontsize=13)

plt.show()
