import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
sns.set_theme()

np.random.seed(0)
N = 20000

W = np.random.normal(50,10,N)
H = W + np.random.normal(30,5,N)
P = 1.85**(W/10.25) - 2*H + np.random.normal(0,5,N)

fig, axs = plt.subplots(1,2, figsize=(15,7), sharex=True, sharey=True)
sns.histplot(x=H, y=P, ax=axs[0])
sns.regplot(x=H, y=P, ax=axs[0], scatter=False)
r, p = pearsonr(H, P)
axs[0].text(.25, .8, 'r={:.2f}, p={:.4f}'.format(r, p),transform=axs[0].transAxes)
axs[0].axis('square')
axs[0].set_xlabel('Health')
axs[0].set_ylabel('Priority Level for Vaccination')
axs[0].set_title(f'Whole Population (N={N})')
yl = (-200,-50)#axs[0].get_ylim()
xl = (20,170)#axs[0].get_xlim()
axs[0].set_xlim(xl)
axs[0].set_ylim(yl)
axs[1].axis('square')
ub = 70
lb = 30
indices_rich = W>=ub
indices_middle = np.logical_and(W>=lb, W<=ub)
indices_poor = W<=lb
sns.histplot(x=H[indices_middle], y=P[indices_middle], ax=axs[1], color='blue')
sns.regplot(x=H[indices_middle], y=P[indices_middle], ax=axs[1], scatter=False, color='blue')
rm, pm = pearsonr(H[indices_middle], P[indices_middle])
axs[1].text(.25, .9, 'r={:.2f}, p={:.4f} Mean Income (blue)'.format(rm, pm),transform=axs[1].transAxes, color='blue')
sns.histplot(x=H[indices_rich], y=P[indices_rich], ax=axs[1], color='red')
sns.regplot(x=H[indices_rich], y=P[indices_rich], ax=axs[1], scatter=False, color='red')
rr, pr = pearsonr(H[indices_rich], P[indices_rich])
axs[1].text(.25, .95, 'r={:.2f}, p={:.4f} Very Wealthy (red)'.format(rr, pr),transform=axs[1].transAxes, color='red')
sns.histplot(x=H[indices_poor], y=P[indices_poor], ax=axs[1], color='green')
sns.regplot(x=H[indices_poor], y=P[indices_poor], ax=axs[1], scatter=False, color='green')
rp, pp = pearsonr(H[indices_poor], P[indices_poor])
axs[1].text(.25, .85, 'r={:.2f}, p={:.4f} Below Average (green)'.format(rp, pp),transform=axs[1].transAxes, color='green')
axs[1].set_xlim(xl)
axs[1].set_ylim(yl)
axs[1].set_title('Visualizing Confounder Wealth via Different Wealth Classes')
# axs[2].axis('square')
# sns.histplot(x=W, y=P, ax=axs[2], color='green')
# sns.regplot(x=W, y=P, ax=axs[2], scatter=False, color='red')
# r, p = pearsonr(W, P)
# axs[2].text(.25, .85, 'r={:.2f}, p={:.4f}'.format(r, p),transform=axs[2].transAxes, color='red')
# axs[2].set_xlim(xl)
# axs[2].set_ylim(yl)
plt.show()