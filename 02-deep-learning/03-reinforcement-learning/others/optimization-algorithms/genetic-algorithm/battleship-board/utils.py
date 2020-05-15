import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns
plt.style.use('seaborn')


def display_heatmap(data, fitness=False, cmap='coolwarm'):

    df_board = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    df_board['ylabel'] = range(10, 0, -1)
    df_board = df_board.set_index('ylabel')

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(df_board, cmap=cmap, linewidths=0.5, annot=True, cbar=False)
    ax.set_xlabel('')
    ax.set_ylabel('')

    if fitness:
        ax.set_title(f'Battleship Board (Fitness: {fitness})')
    else:
        ax.set_title('Battleship Board')
        
    plt.show()

def plot_population_stats(population):

    fig, ay = plt.subplots(1, 1, figsize = (10, 8))
    ay = sns.lineplot(x='Generation', y='Fitness', data=population)
    ay.set_title('Collective Population Statistics')
    
    plt.savefig('./images/plot_population_stats.png')
    
    plt.show()
    