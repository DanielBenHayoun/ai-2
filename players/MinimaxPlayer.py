"""
MiniMax Player
"""
from utils import get_directions
import numpy as np
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import MiniMax,State
import time
from copy import deepcopy

PLAYER = 1
RIVAL = 2
BLOCKED = -1
FREE_PATH = 0
#################
RIGHT = 1
DOWN = 2
LEFT = 3
UP = 4
#################
X = 0
Y = 1
#################
class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
       
        #self.penalty_score = penalty_score
        self.minimax = MiniMax(utility, succ, None, goal=is_goal_state)
        self.fruits_poses = None
        self.fruits_on_board_dict = {}
        self.locations = [None, None, None]

    def set_game_params(self, board):
        """Set the game parameters needed for this player.
        This function is called before the game starts.
        (See GameWrapper.py for more info where it is called)
        input:
            - board: np.array, a 2D matrix of the board.
        No output is expected.
        """
        self.board = board
        self.n_rows = len(self.board[0])  # cols number
        self.n_cols = len(self.board)     # rows number
        player_pos = np.where(board == 1)
        rival_pos = np.where(board == 2)
        fruits_poses = np.where(board > 2)
        self.locations[PLAYER] = tuple(ax[0] for ax in player_pos)
        self.locations[RIVAL] = tuple(ax[0] for ax in rival_pos)
        if len(fruits_poses) > 0 and len(fruits_poses[0]) > 0:
            self.fruits_poses = tuple(ax[i] for ax in fruits_poses for i in range(len(fruits_poses[0])))
        

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        
        start_time = time.time()

        d = 1  
        # Make the initial state:
        state = State()
        state.update_state(self.board,self.locations,self.fruits_poses,PLAYER)
        
        _, direction = self.minimax.search(state,d,True,time_limit)

        prev_time = time.time() - start_time
        next_time = self.estimate_next_time(prev_time)
        time_until_now = time.time() - start_time

        while time_until_now + next_time < time_limit and not direction:
            iteration_start_time = time.time()
            d += 1
            _, direction = self.minimax.search(state,d,True,time_limit)
            prev_time = time.time() - iteration_start_time
            next_time = self.estimate_next_time(prev_time)
            time_until_now = time.time() - start_time

        
        self.board[self.locations[PLAYER]] = -1

        if direction is None:
            exit()
        
        best_new_location = (self.locations[PLAYER][X] + direction[X], self.locations[PLAYER][Y] + direction[Y])
        self.board[best_new_location] = 1
        self.locations[PLAYER] = best_new_location

        return direction


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.board[self.locations[RIVAL]] = -1
        self.locations[RIVAL] = pos
        self.board[pos] = 2


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.fruits_on_board_dict = deepcopy(fruits_on_board_dict)
            

    ########## helper functions in class ##########
    def estimate_next_time(self,cu_time:float) -> float:
        return cu_time

    ########## helper functions for MiniMax algorithm ##########


    # def reachable_white_squares(self, player_type: str) -> int:
    #     """Get the total number of reachable squares from the current player location in the game's board
    #     input:
    #         - player_type:  str, one of the values ['player','rival']
    #     output:
    #         Returns the len of reachable_squares list.
    #         This value presents the total number of reachable squares from the current player location in the game's board
    #     """
    #     initial_player_location = self.get_player_location(player_type)

    #     row_min = 0
    #     row_max = self.n_rows - 1
    #     col_min = 0
    #     col_max = self.n_culs - 1

    #     reachable_board = np.zeros((self.n_rows, self.n_culs))
    #     reachable_squares = list()
    #     reachable_squares.append(initial_player_location)
    #     start_index = 0
    #     len_reachable_squares = 1

    #     # adds reachable squares to reachable_squares list
    #     while start_index < len(reachable_squares):
    #         player_location = reachable_squares[start_index]
    #         # TODO change to keyboard_directions
    #         for d in self.directions:
    #             i = player_location[0] + d[0]
    #             j = player_location[1] + d[1]

    #             # then move is legal
    #             if row_min <= i <= row_max and col_min <= j <= col_max and self.board[i][j] == 0:
    #                 new_location = (i, j)
    #                 # the square in new_loc is available in the game's board
    #                 if self.board[i][j] == 0 and (not reachable_board[i][j]):
    #                     reachable_board[i][j] = 1
    #                     # HERE WE CHANGE THE LENGTH OF reachable_squares
    #                     reachable_squares.append(new_location)
    #                     # len_reachable_squares += 1
    #         start_index += 1
    #     # Returns the len of reachable_squares list.
    #     # This value presents the total number of reachable squares from the current player location in the game's board
    #     return len_reachable_squares - 1

def legal_move(board,i,j):
    return 0 <= i < len(board) and 0 <= j < len(board[0]) and board[i][j] not in [-1, 1, 2]

def succ(state:State) -> State : 
    
    try:
        for d in get_directions():
            child = deepcopy(state)
            player_type = child.player_type
            player_location = child.locations[player_type] 

            i = child.locations[player_type][X] + d[X]
            j = child.locations[player_type][Y] + d[Y]
            # then move is legal
            if legal_move(child.board,i,j):
                new_location = (i, j)
                child.board[player_location] = -1
                child.board[new_location] = player_type
                child.locations[player_type] = new_location
                child.dir = d
                yield child # Yield a new state that made the move
        
    except StopIteration:
        return None

def is_stuck(state:State,player_type):
        for d in get_directions():
            i = state.locations[player_type][X] + d[X]
            j = state.locations[player_type][Y] + d[Y]
            if legal_move(state.board,i,j):
                return False
        return True

def is_goal_state(state:State):
    
    is_player_stuck = is_stuck(state, PLAYER)
    is_rival_stuck = is_stuck(state, RIVAL)
    return is_player_stuck or is_rival_stuck
    # TODO make sure losing and wining score are reasonble
    if state.player_type == PLAYER:
        if is_player_stuck and (not is_rival_stuck):
            return -10000000, 1, True
        elif is_player_stuck and is_rival_stuck:
            return 0, 1, True

    else:
        if is_rival_stuck and (not is_player_stuck):
            return 10000000, 1, True
        elif is_rival_stuck and is_player_stuck:
            return 0, 1, True

def utility(state:State,maximizing_player):
    """
    utility and heuristic function
    """
    player_type = PLAYER if maximizing_player else RIVAL
    #candidates = []
    best_move_score = -1
    best_move = None
    for d in get_directions():
            i = state.locations[player_type][X] + d[X]
            j = state.locations[player_type][Y] + d[Y]
            if legal_move(state.board,i,j):
                pos=(i, j)
                ############### Huristic ###############
                score = 0
                num_steps_available = 0
                for d in get_directions():
                    i = pos[0] + d[0]
                    j = pos[1] + d[1]

                    # Check legal moves
                    if legal_move(state.board,i,j):
                        num_steps_available += 1

                if num_steps_available == 0:
                    score -= 1
                else:
                    score =+ 4 - num_steps_available
                # Check for fruit:
                if state.board[pos[0]][pos[1]] > 2:
                    score += 2
                ######################################
                if score > best_move_score:
                    best_move, best_move_score = d, score
                #candidates.append(d)
    
    return best_move_score,best_move

