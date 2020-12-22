"""Search Algos: MiniMax, AlphaBeta
"""
#from players.MinimaxPlayer import Player
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
# TODO: you can import more modules, if needed
from typing import Callable
##
## A State class
class State:
    def __init__(self,board=None,fruits=None):
        self.player_type = 1 # Player, 2 Rival
        self.board = board
        self.locations = [None, None, None]
        self.dir = dir
        self.fruits = fruits
        if self.board == None:
            self.board = []
        if self.fruits == None:
            self.fruits = []

    def set_location(self,pos):
        self.locations[self.player_type] = pos
    
    def update_state(self,board,locations=None,fruits=None,player_type=None):
        self.board = board
        if locations != None:
            self.locations = locations
        if fruits != None:
            self.fruits = fruits
        if player_type != None:
            self.player_type = player_type

    def set_player(self,player_type):
        self.player_type == 1 if player_type else 2 

############################################

class SearchAlgos:
    def __init__(self, utility, succ:Callable[[State],State], perform_move, goal=None):
        """The constructor for all the search algos.
        You can code these functions as you like to, 
        and use them in MiniMax and AlphaBeta algos as learned in class
        :param utility: The utility function.
        :param succ: The succesor function.
        :param perform_move: The perform move function.
        :param goal: function that check if you are in a goal state.
        """
        self.utility = utility
        self.succ = succ
        self.perform_move = perform_move
        self.goal = goal

    def search(self, state, depth, maximizing_player):
        pass


class MiniMax(SearchAlgos):

    def search(self, state:State, depth, maximizing_player,time_limit):
        """Start the MiniMax algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        state.set_player(maximizing_player)
        # Check base cases
        is_goal = self.goal(state)
        if is_goal or depth == 0:
            return self.utility(state,maximizing_player)
        
        # Else, Search:

        # Max Node:
        if maximizing_player:
            cur_max = float('-inf')
            cur_direction = None

            for child_state in self.succ(state):
                # child_state: (board,direction)
                minmax_value, _ = self.search(child_state,depth-1, not maximizing_player,time_limit)
                cur_max = max(minmax_value, cur_max)
                if cur_max == minmax_value:
                    cur_direction = child_state.dir # Move in the best direction
                # Preform move?

            return cur_max, cur_direction
        # Min Node:
        else:  
            cur_min = float('inf')
            
            for child_state in self.succ(state):
                minmax_value, _ = self.search(child_state,depth-1, not maximizing_player,time_limit)
                cur_min = min(minmax_value, cur_min)

            return cur_min, None



class AlphaBeta(SearchAlgos):

    def search(self, state, depth, maximizing_player, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError
