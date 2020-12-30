"""Search Algos: MiniMax, AlphaBeta
"""
#from players.MinimaxPlayer import Player
from copy import deepcopy
from os import stat
import time

import numpy as np
from utils import ALPHA_VALUE_INIT, BETA_VALUE_INIT
from typing import Callable
##
## A State class
class State:
    
    def __init__(self,directions,board,locations,fruits_dict,player_type,players_score,penalty_score,fruits_ttl,turns):
        self.directions = deepcopy(directions)
        self.X=0
        self.Y=1
        self.player_type = player_type
        self.board = deepcopy(board)
        self.locations = locations
        self.fruits_dict = deepcopy(fruits_dict)
        self.fruits_ttl = fruits_ttl
        self.dir = dir
        self.penalty_score = penalty_score
        self.players_score=[None, None, None]
        self.players_score[1] = players_score[0]
        self.players_score[2] = players_score[1]

        #self.sp_points = 0 # Simple Player points

        total_whites = len(np.where(board == 0)[0])
        self.max_turns = total_whites // 2
        self.turns = turns
     

    def set_location(self,player_type,pos):
        self.locations[player_type] = pos
    
    def get_location(self):
        return self.locations[self.player_type]

    def set_player(self,player_type):
        self.player_type = 1 if player_type else 2 

    def change_player(self):
        if self.player_type == 1:
            self.player_type = 2
        else:
            self.player_type = 1

    def legal_move(self,i,j):
        return 0 <= i < len(self.board) and 0 <= j < len(self.board[0]) and self.board[int(i)][int(j)] not in [-1, 1, 2]

    def add_direction(self, player_type,d):
        return (self.locations[player_type][0] + d[0],self.locations[player_type][1] + d[1])
    
    def available_steps(self, location):
        available_steps = 0
        for direction in self.directions:
            i, j = location[0]+direction[0],location[1]+direction[1]
            if self.legal_move(int(i),int(j)):
                available_steps += 1
        return available_steps

    def half_game(self):
        return self.turns >= (self.max_turns // 2)

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
        start_time = time.time()
        #state.set_player(maximizing_player)
        # Check base cases
        is_goal = self.goal(state)
        if is_goal or depth == 0:
            return self.utility(state,maximizing_player),state.dir,is_goal,state
        
        # Max Node:
        if maximizing_player:
            cur_max = float('-inf')
            cur_direction = None
            curr_reach_the_end = False
            chosen_state = None

            for child_state in self.succ(state):
                iter_time = time.time() - start_time
                #print(f'Passed: {time_until_now+iter_time}')
                if iter_time >=  time_limit:
                    #print(f'Passed {iter_time} / {time_limit}')
                    raise Exception("Time's up!")
                minmax_value, _, reach_the_end,curr_state = self.search(child_state,depth-1, not maximizing_player,time_limit-iter_time)
                cur_max = max(minmax_value, cur_max)
                if cur_max == minmax_value:
                    cur_direction = child_state.dir # Move in the best direction
                    curr_reach_the_end = reach_the_end 
                    chosen_state = curr_state
                #print(f'{minmax_value}',end='|')
                
            #print("")
            return cur_max, cur_direction ,curr_reach_the_end,chosen_state
        # Min Node:
        else:  
            cur_min = float('inf')
            curr_reach_the_end = False
            chosen_state = None

            for child_state in self.succ(state):
                iter_time = time.time() - start_time
                if iter_time >= time_limit:
                    #print(f'Passed {iter_time} / {time_limit}')
                    raise Exception("Time's up!")
                minmax_value, _, reach_the_end,curr_state = self.search(child_state,depth-1, not maximizing_player,time_limit-iter_time)
                cur_min = min(minmax_value, cur_min)
                if cur_min == minmax_value:
                    curr_reach_the_end = reach_the_end 
                    chosen_state = curr_state

            return cur_min, None,curr_reach_the_end,chosen_state



class AlphaBeta(SearchAlgos):

    def search(self, state:State, depth, maximizing_player,time_limit, alpha=ALPHA_VALUE_INIT, beta=BETA_VALUE_INIT):
        """Start the AlphaBeta algorithm.
        :param state: The state to start from.
        :param depth: The maximum allowed depth for the algorithm.
        :param maximizing_player: Whether this is a max node (True) or a min node (False).
        :param alpha: alpha value
        :param: beta: beta value
        :return: A tuple: (The min max algorithm value, The direction in case of max node or None in min mode)
        """
        start_time = time.time()
        #state.set_player(maximizing_player)
        # Check base cases
        is_goal = self.goal(state)
        if is_goal or depth == 0:
            return self.utility(state,maximizing_player),state.dir,is_goal,state
        
        # Max Node:
        if maximizing_player:
            cur_max = float('-inf')
            cur_direction = None
            curr_reach_the_end = False
            chosen_state = None

            for child_state in self.succ(state):
                iter_time = time.time() - start_time
                #print(f'Passed: {time_until_now+iter_time}')
                if iter_time >=  time_limit:
                    #print(f'Passed {iter_time} / {time_limit}')
                    raise Exception("Time's up!")
                minmax_value, direction, reach_the_end,curr_state = self.search(child_state,depth-1, not maximizing_player,time_limit-iter_time,alpha,beta)
                cur_max = max(minmax_value, cur_max)
                if cur_max == minmax_value:
                    cur_direction = child_state.dir # Move in the best direction
                    curr_reach_the_end = reach_the_end 
                    chosen_state = curr_state
                #print(f'{minmax_value}',end='|')
                alpha = max(cur_max, alpha)
                if cur_max >= beta: # Chop chop
                    return float('inf'), direction, False , chosen_state
                
            #print("")
            return cur_max, cur_direction ,curr_reach_the_end,chosen_state
        # Min Node:
        else:  
            cur_min = float('inf')
            curr_reach_the_end = False
            chosen_state = None

            for child_state in self.succ(state):
                iter_time = time.time() - start_time
                if iter_time >= time_limit:
                    #print(f'Passed {iter_time} / {time_limit}')
                    raise Exception("Time's up!")
                minmax_value, direction, reach_the_end,curr_state = self.search(child_state,depth-1, not maximizing_player,time_limit-iter_time,alpha,beta)
                cur_min = min(minmax_value, cur_min)
                if cur_min == minmax_value:
                    curr_reach_the_end = reach_the_end 
                    chosen_state = curr_state
                beta = min(cur_min, beta)
                if cur_min <= alpha: # Chop chop
                    return float('-inf'), direction, False, chosen_state
            return cur_min, None,curr_reach_the_end,chosen_state
