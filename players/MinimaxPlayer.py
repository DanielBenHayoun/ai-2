"""
MiniMax Player
"""
import numpy as np
from players.AbstractPlayer import AbstractPlayer
from SearchAlgos import MiniMax
import time


class Player(AbstractPlayer):
    def __init__(self, game_time, penalty_score):
        # keep the inheritance of the parent's (AbstractPlayer) __init__()
        AbstractPlayer.__init__(self, game_time, penalty_score)
        # TODO: initialize more fields, if needed, and the Minimax algorithm from SearchAlgos.py
        self.keyboard_direction = {
            'w': self.directions[1],
            'a': self.directions[2],
            's': self.directions[3],
            'd': self.directions[0]

        }

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
        self.n_culs = len(self.board)     # rows number
        for i, row in enumerate(board):
            for j, val in enumerate(row):
                if val == 1:
                    self.player_location = (i, j)
                    break
                elif val == 2:
                    self.rival_location = (i, j)
                    break
                elif val > 2:
                    self.fruits[i][j] = val

    def make_move(self, time_limit, players_score):
        """Make move with this Player.
        input:
            - time_limit: float, time limit for a single turn.
        output:
            - direction: tuple, specifing the Player's movement, chosen from self.directions
        """
        # TODO call minimax with max-value and time limit and the current board as starting state
        # make sure the move is legal
        # update board locations
        # return self.keyboard_direction[]
        # check if onw of the players is stuck
        # check time limit to send for alg.
        # utility=?

        # minimax=MiniMax()

    def set_rival_move(self, pos):
        """Update your info, given the new position of the rival.
        input:
            - pos: tuple, the new position of the rival.
        No output is expected
        """
        # TODO: erase the following line and implement this function.
        raise NotImplementedError

    def update_fruits(self, fruits_on_board_dict):
        """Update your info on the current fruits on board (if needed).
        input:
            - fruits_on_board_dict: dict of {pos: value}
                                    where 'pos' is a tuple describing the fruit's position on board,
                                    'value' is the value of this fruit.
        No output is expected.
        """
        for pos, value in fruits_on_board_dict.items():
            i, j = pos
            self.fruits[i][j] = value

    ########## helper functions in class ##########
    def get_player_location(self, player_type: str):
        """Get player location
        input:
            - player_type: str, one of the values ['player','rival']
        output:
            - pos: tuple, position of the player
        """
        if player_type == 'player':
            return self.player_location
        return self.rival_location

    def is_stuck(self, player_type: str):
        """
        input:
            - player_type: str, one of the values ['player','rival']
        output:
            -  True if player_type is stuck. else, returns False
        """
        for key, move in self.keyboard_direction:
            location = self.get_player_location(player_type)
            i = location[0] + move[0]
            j = location[1] + move[1]
            # then move is legal
            if 0 <= i < self.n_culs and 0 <= j < self.n_rows and self.board[i][j] == 0:
                return False
        return True

    ########## helper functions for MiniMax algorithm ##########

    def succ(self, current_board,player_type: str) -> list:
        """Get succesors list
        input:
            - player_type:  str, one of the values ['player','rival']
            - current_board: 
        output:
            -  list of successor states to the given board_state
        """
        player_location = self.get_player_location(player_type)
        try:
            for key, move in self.keyboard_direction:
                i = player_location[0] + move[0]
                j = player_location[1] + move[1]
                if 0 <= i < self.n_culs and 0 <= j < self.n_rows and current_board[i][j] == 0:
                    new_location = (i, j)
                    current_board[player_location] = -1
                    current_board[new_location] = player_type
                    if player_type == 1:
                        self.player_location = new_location
                    else:
                        self.rival_location = new_location
                    yield current_board, move
        except StopIteration:
            return None

    def reachable_white_squares(self, player_type: str) -> int:
        """Get the total number of reachable squares from the current player location in the game's board
        input:
            - player_type:  str, one of the values ['player','rival']
        output:
            Returns the len of reachable_squares list.
            This value presents the total number of reachable squares from the current player location in the game's board
        """
        initial_player_location = self.get_player_location(player_type)

        row_min = 0
        row_max = self.n_rows - 1
        col_min = 0
        col_max = self.n_culs - 1

        reachable_board = np.zeros((self.n_rows, self.n_culs))
        reachable_squares = list()
        reachable_squares.append(initial_player_location)
        start_index = 0
        len_reachable_squares = 1

        # adds reachable squares to reachable_squares list
        while start_index < len(reachable_squares):
            player_location = reachable_squares[start_index]
            # TODO change to keyboard_directions
            for d in self.directions:
                i = player_location[0] + d[0]
                j = player_location[1] + d[1]

                # then move is legal
                if row_min <= i <= row_max and col_min <= j <= col_max and self.board[i][j] == 0:
                    new_location = (i, j)
                    # the square in new_loc is available in the game's board
                    if self.board[i][j] == 0 and (not reachable_board[i][j]):
                        reachable_board[i][j] = 1
                        # HERE WE CHANGE THE LENGTH OF reachable_squares
                        reachable_squares.append(new_location)
                        # len_reachable_squares += 1
            start_index += 1
        # Returns the len of reachable_squares list.
        # This value presents the total number of reachable squares from the current player location in the game's board
        return len_reachable_squares - 1

    def is_goal_state(self,board_state, player_type: str):
        """check whether board_state is a goal state
        input:
            - player_type:  str, one of the values ['player','rival']
            - board_state: 
        output:
            tuple which contains the minimax value of board_state, and the number of expanded leaves in this iteration
        """

        is_player_stuck = self.is_stuck(board_state, 'player')
        is_rival_stuck = self.is_stuck(board_state, 'rival')
        # TODO make sure losing and wining score are reasonble
        if player_type == 'player':
            if is_player_stuck and (not is_rival_stuck):
                return -10000000, 1, True
            elif is_player_stuck and is_rival_stuck:
                return 0, 1, True

        else:
            if is_rival_stuck and (not is_player_stuck):
                return 10000000, 1, True
            elif is_rival_stuck and is_player_stuck:
                return 0, 1, True
