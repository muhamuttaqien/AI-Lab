import pandas as pd
import matplotlib.pyplot as plt

import seaborn as sns


def display_heatmap(data):

    df_board = pd.DataFrame(data, columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'])
    df_board['ylabel'] = range(10, 0, -1)
    df_board = df_board.set_index('ylabel')

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    sns.heatmap(df_board, cmap='coolwarm', linewidths=0.5, annot=True, cbar=False)
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_title('Battleship Board')

    plt.show()
    