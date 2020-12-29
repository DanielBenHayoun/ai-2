"""
MiniMax Player with AlphaBeta pruning
"""
from utils import get_directions
import numpy as np
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import AlphaBeta,State
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
       
        self.penalty_score = penalty_score
        self.alphabeta = AlphaBeta(utility, succ, None, goal=is_goal_state)
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
        self.board = board.copy()
        self.n_rows = len(self.board[0])  # cols number
        self.n_cols = len(self.board)     # rows number
        self.fruits_ttl = min(self.n_rows,self.n_cols)+1
        player_pos = np.where(board == 1)
        rival_pos = np.where(board == 2)
        self.locations[PLAYER] = tuple(ax[0] for ax in player_pos)
        self.locations[RIVAL] = tuple(ax[0] for ax in rival_pos)
        self.turns = 0
        # if len(fruits_poses) > 0 and len(fruits_poses[0]) > 0:
        #     self.fruits_poses = tuple(ax[i] for ax in fruits_poses for i in range(len(fruits_poses[0])))
        # num_free_places = len(np.where(self.map == 0)[0])
        
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
        
        reach_the_end = False
        best_direction = None
        chosen_state = None
        if time_limit >= 5:
            TIME_ESTIMATION = 0.9 
        else:
            TIME_ESTIMATION = 0.85

        while not reach_the_end: # and d < len(state.board)*len(state.board[0]):
            
            iter_time_limit = TIME_ESTIMATION * ( time_limit - (time.time() - start_time) )
            state = State(get_directions(),self.board,self.locations,self.fruits_on_board_dict,PLAYER,players_score,self.penalty_score,self.fruits_ttl)

            try:
                _, best_direction, reach_the_end,chosen_state = self.alphabeta.search(state,d,True,iter_time_limit,alpha=float('-inf'), beta=float('inf'))
                d += 1
                print(f'AB>>{d}')# TODO: REMOVE
            except Exception as e:
                print(f'AB::{e}') # TODO: REMOVE
                break
            
        # Set new location       
        if best_direction == None:
            best_direction = self.get_random_move() # TODO: Dirty fix
        self.set_player_location(best_direction)
        
        self.turns += 1
        return best_direction


    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        self.board[self.locations[RIVAL]] = -1
        self.locations[RIVAL] = pos
        self.board[pos] = 2
        #  Check for fruit:
        if pos in self.fruits_on_board_dict.keys():
            self.fruits_on_board_dict.pop(pos)


    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        self.fruits_on_board_dict = deepcopy(fruits_on_board_dict)

        if self.fruits_ttl <= 0:
            return
        self.fruits_ttl -= 1
        # Remove all fruits if their TTL expired
        if self.fruits_ttl <= 0: #TODO: Check it works (mask)
            mask = self.board>2
            self.board[mask] = 0
            #self.board = np.array([[0 if i not in [0, 1, 2, -1] else i for i in line] for line in self.board])
            
            

    ########## helper functions in class ##########
    def estimate_next_time(self,cu_time:float) -> float:
        return cu_time

    def set_player_location(self, best_direction):
        self.board[self.locations[PLAYER]] = -1     
        best_new_location = (self.locations[PLAYER][X] + best_direction[X], self.locations[PLAYER][Y] + best_direction[Y])
        self.board[best_new_location] = 1
        self.locations[PLAYER] = best_new_location
        #  Check for fruit:
        if best_new_location in self.fruits_on_board_dict.keys():
            self.fruits_on_board_dict.pop(best_new_location)
    
    def get_random_move(self):
        for d in self.directions:
            (i,j) = self.locations[PLAYER]
            if 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and self.board[i][j] not in [-1, 1, 2]:
                return d
        return (-1,-1)

    ########## helper functions for MiniMax algorithm ##########


def reachable_white_squares(board,pos) -> int:
    #(i,j) = pos
    initial_location = pos

    reachable_board = np.zeros((len(board), len(board[0])))
    reachable_squares = list()
    reachable_squares.append(initial_location)
    start_index = 0
    found_reachable = True

    while found_reachable and start_index < len(reachable_squares):
        player_location = reachable_squares[start_index]
        found_reachable = False
        for d in get_directions():
            i = player_location[0] + d[0]
            j = player_location[1] + d[1]
            if legal_move(board,i,j):
                new_location = (i, j)
                if not reachable_board[i][j]:
                    reachable_board[i][j] = 1
                    reachable_squares.append(new_location)
                    found_reachable = True
        start_index += 1

    return len(reachable_squares) - 1

def legal_move(board,i,j):
    return 0 <= i < len(board) and 0 <= j < len(board[0]) and (board[i][j] not in [-1, 1, 2])

def succ(state:State) -> State : 
    sp_move = state.computeSimplePlayerSteps()

    try:
        for d in state.directions:
            i = state.locations[state.player_type][X] + d[X]
            j = state.locations[state.player_type][Y] + d[Y]
            # Check children states:
            if legal_move(state.board,i,j):
                child = deepcopy(state)
                child.change_player()
                child.dir = d
                
                if state.player_type == PLAYER:
                    child.fruits_ttl -= 1
                # Determind locations
                curr_location = child.locations[state.player_type] 
                new_location = (curr_location[X] + d[X] , curr_location[Y] + d[Y])
                # Mind the fruits
                if new_location in state.fruits_dict.keys():
                    child.players_score[child.player_type] += child.board[new_location]
                    child.fruits_dict.pop(new_location)
                # TODO add penalty ?
                # Move the player
                child.board[curr_location] = -1
                child.board[new_location] = state.player_type
                child.locations[state.player_type] = new_location
                # Consider Simple Player move:
                #child.change_player()
                if child.player_type == PLAYER and child.available_steps(new_location) == sp_move:
                    child.sp_points +=  ((4-sp_move)/10.0) + 1
                
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

    is_player_stuck = is_stuck(state, state.player_type)
    return is_player_stuck
    # is_rival_stuck = is_stuck(state, RIVAL)
    # if is_rival_stuck and state.player_type == PLAYER:
    #     return False
    # return is_player_stuck or is_rival_stuck

def Manhattan(start, end):
    return abs(start[1] - end[1]) + abs(start[0] - end[0])

def utility(state:State,maximizing_player):
    """
    utility and heuristic function
    """
    #######################[Goal]#########################
    is_current_player_stuck = is_stuck(state,state.player_type)
    other_player = RIVAL if state.player_type == PLAYER else PLAYER
    is_other_player_stuck = is_stuck(state,other_player)
    # Check if stuck
    if is_current_player_stuck:
        if is_other_player_stuck:
            return 10000000 if state.players_score[state.player_type] > state.players_score[other_player] else -10000000
        # Else, add penalty
        state.players_score[state.player_type] -= state.penalty_score
        return 10000000 if state.players_score[state.player_type] > state.players_score[other_player] else -10000000
    ######################################################
    # Else
    best_move_score = -1
    h1 = state.sp_points#/10 #4 - state.available_steps(state.player_type) 
    #h1 = 4 - state.available_steps(state.get_location())
    h2 = -1
    if state.fruits_ttl > 0 and len(state.fruits_dict) > 0:
        min_fruit_dist = float('inf')
        for fruit_loc in state.fruits_dict:
            curr_fruit_dist = Manhattan(state.locations[state.player_type], fruit_loc)
            # Check what is the closest fruit reachable
            if curr_fruit_dist < min_fruit_dist and curr_fruit_dist <= state.fruits_ttl:
                other_player_fruit_dist = Manhattan(state.locations[other_player], fruit_loc)
                if curr_fruit_dist < other_player_fruit_dist:
                    min_fruit_dist = curr_fruit_dist
        max_dist = len(state.board)+len(state.board[0])
        h2 = (10.0/min_fruit_dist)+1 if min_fruit_dist < float('inf') else -1
    w = 0.7 if h2 > 0 else 1
    best_move_score = w*h1 + (1-w)*h2 
    best_move_score += state.players_score[state.player_type]
    # if h2>0 and state.player_type == PLAYER:
    #     print(f'{state.fruits_ttl}:| [h1: {h1}] | [h2: {h2}] | [score: {best_move_score}] | w is {w}')
    return best_move_score

