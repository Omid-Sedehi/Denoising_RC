import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

Data_GWN = np.load('Error_Gain_GWN.npz')
Data_Pink = np.load('Error_Gain_Pink.npz')
Data_Violet = np.load('Error_Gain_Violet.npz')

# Create violin plot
plt.figure(figsize=(5, 4))
plt.style.use('./Styles/AIPStyles.mplstyle')
ax = sns.violinplot(data=[Data_GWN["Gain"].flatten(), Data_Violet["Gain"].flatten(), Data_Pink["Gain"].flatten()],
                    palette=["white", "violet", "pink"], inner="quartile")

# Add individual data points with jitter
# for i, noise_type in enumerate(["White Noise", "Pink Noise", "Brown Noise"]):
num_reps = 20
jitter = np.random.normal(0, 0.05, num_reps)  # Small jitter for better visibility
ax.scatter( jitter, Data_GWN["Gain"].flatten(), color='white', edgecolor='black', alpha=0.6, s=25, zorder=2)
ax.scatter( 1+jitter, Data_Violet["Gain"].flatten(), color='violet', edgecolor='black', alpha=0.6, s=25, zorder=2)
ax.scatter( 2+jitter, Data_Pink["Gain"].flatten(), color='pink', edgecolor='black', alpha=0.6, s=25, zorder=2)

plt.xticks([0, 1, 2], ["Gaussian White Noise", "Violet Noise", "Pink Noise"])
plt.ylabel("Denoising Gain")
plt.xlabel("Different Types of Noise")
# plt.title("Denoising Gain Distribution")
# plt.ylim([0,30])
plt.yscale('log')
plt.show()

