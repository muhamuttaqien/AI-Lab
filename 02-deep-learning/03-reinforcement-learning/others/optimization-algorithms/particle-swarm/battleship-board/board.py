import numpy as np


class Battleship(object):
    """Generate battleship board and solution."""
    
    def __init__(self, dimension=10):
        
        self.dimension = dimension
        self.board = np.zeros((dimension, dimension), dtype=int)
        self.ship_lengths = [5, 4, 3, 2, 2]
        
    def create_board(self):
        
        # for each ship
        for ship_length in self.ship_lengths:
            
            is_ship_placed = False
            
            while not is_ship_placed:
                
                # seed a coordinate for the head of a ship
                head = tuple(np.random.randint(self.dimension, size=2))
                
                # choose a direction for the ship to be laid out
                heading = np.random.randint(4)
                
                # check that the ship does not hang off the edge of the board
                if heading == 0:
                    tail = (head[0] - ship_length + 1, head[1])
                elif heading == 1:
                    tail = (head[0] + ship_length - 1, head[1])
                elif heading == 2:
                    tail = (head[0], head[1] + ship_length - 1)
                elif heading == 3:
                    tail = (head[0], head[1] - ship_length + 1)
                    
                if not ((0 <= tail[0] <= self.dimension-1) and (0 <= tail[1] <= self.dimension-1)):
                    continue

                # check that the ship does not overlap with any others
                NS_min = min(head[0], tail[0])
                NS_max = max(head[0], tail[0])
                EW_min = min(head[1], tail[1])
                EW_max = max(head[1], tail[1])

                if sum(sum(self.board[NS_min:NS_max+1, EW_min:EW_max+1])) != 0:
                    continue

                # place the ship
                self.board[NS_min:NS_max+1, EW_min:EW_max+1] = 1
                is_ship_placed = True
            
        # check number of pieces on the board
        if sum(self.ship_lengths) == sum(sum(self.board)):
            print('Correct number of pieces on board.')
        else:
            print('Incoorect number of pieces on board.')
            
        # represent board solution in genetic form
        genetic_solution = ''.join(str(x) for x in list(self.board.flatten()))
        
        return self.board, genetic_solution