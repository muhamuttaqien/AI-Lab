import numpy as np
import matplotlib.pyplot as plt

def plot_values(V, name='State', shape=(4,4)):
    
    # reshape value function
    V_sq = np.reshape(V, shape)
    
    # plot the state-value function
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    
    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5), ha='center', va='center', fontsize=14)
    plt.tick_params(bottom=False, left=False, labelbottom=False, labelleft=False)
    plt.title(f'{name} Function')
    
    plt.show()
