import numpy as np

# global variables
BOARD_ROWS = 3
BOARD_COLS = 4
WIN_STATE = (0, 3)
LOSE_STATE = (1, 3)
START = (2, 0)
DETERMINISTIC = True

class State:
    def __init__(self, state=START,win = WIN_STATE,lose = LOSE_STATE):
        self.board = np.zeros([BOARD_ROWS, BOARD_COLS])
        self.state = state
        self.isEnd = False
        self.determine = DETERMINISTIC
        self.win = WIN_STATE
        self.lose = LOSE_STATE

    def giveReward(self,state):
        if state == WIN_STATE:
            return 1
        elif state == LOSE_STATE:
            return -1
        else:
            return -0.04

    def isEndFunc(self):
        if (self.state == WIN_STATE) or (self.state == LOSE_STATE):
            self.isEnd = True

    def nxtPosition(self, state,action):
        """
        action: up, down, left, right
        -------------
        0 | 1 | 2| 3|
        1 |
        2 |
        return next position
        """
        if self.determine:
            if action == "up":
                nxtState = (state[0] - 1, state[1])
            elif action == "down":
                nxtState = (state[0] + 1, state[1])
            elif action == "left":
                nxtState = (state[0], state[1] - 1)
            else:
                nxtState = (state[0], state[1] + 1)
            # if next state legal
            if (nxtState[0] >= 0) and (nxtState[0] <= (BOARD_ROWS -1)):
                if (nxtState[1] >= 0) and (nxtState[1] <= (BOARD_COLS -1)):
                    if nxtState != (1, 1):
                        return nxtState
            return state

    def showBoard(self):
        self.board[self.state] = 1
        self.board[self.win] = -1
        self.board[self.lose] = -1
        for i in range(0, BOARD_ROWS):
            print('-----------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                if self.board[i, j] == 1:
                    token = '*'
                if self.board[i, j] == -1:
                    token = 'z'
                if self.board[i, j] == 0:
                    token = '0'
                out += token + ' | '
            print(out)
        print('-----------------')


class Agent:

    def __init__(self):
        self.states = []
        self.actions = ["up", "down", "left", "right"]
        self.State = State()
        self.gam = 0.9
        self.exp_rate = 0.3

        # initial state reward
        self.state_values = {}
        for i in range(BOARD_ROWS):
            for j in range(BOARD_COLS):
                self.state_values[(i, j)] = 0  # set initial value to 0
        
        self.state_values[WIN_STATE] = 1
        self.state_values[LOSE_STATE] = -1
    def reset(self):
        self.states = []
        self.State = State()


    def get_rightangle(self,action):
        if action == 'up' or action == 'down':
            return ['left','right']
        return ['up','down']
    def get_reward(self, state,action):
        
        reward = 0
        per_pend = self.get_rightangle(action)
        
        intented_state = self.State.nxtPosition(state,action)
        for i in per_pend:
            reward = reward + 0.1*self.State.giveReward(self.State.nxtPosition(state,i)) + (0.1* self.gam )* self.state_values[self.State.nxtPosition(state,i)]
        
        reward = reward + 0.8*self.State.giveReward(intented_state) + (0.8*self.gam)*self.state_values[intented_state]

        return reward


        

    def play(self, rounds=10):
        i = 0
        delta = 1
        while True:
            error = 0
            for i in range(0, BOARD_ROWS):
                for j in range(0, BOARD_COLS):
                    if (i,j) == WIN_STATE or (i,j) == LOSE_STATE:
                        continue
                    action_value = 0
                    max_value = -np.inf
                    for k in self.actions:
                        reward = self.get_reward((i, j),k)
                        action_value = (0.25)*reward
                        max_value = max(max_value, action_value)
                        error = max(error,abs(self.state_values[(i, j)] - max_value))
                    self.state_values[(i, j)] = max_value
            if error < delta:
                break
    def showValues(self):
        for i in range(0, BOARD_ROWS):
            print('----------------------------------')
            out = '| '
            for j in range(0, BOARD_COLS):
                out += str(self.state_values[(i, j)]).ljust(4) + ' | '
            print(out)
        print('----------------------------------')
if __name__ == "__main__":
    s = State()
    s.showBoard()
    ag = Agent()
    ag.play()
    ag.showValues()
