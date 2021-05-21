import numpy as np
from collections import defaultdict

####
import gym
import gym_chess
import random
import chess
import chess.engine
from gym_chess.alphazero import BoardEncoding

from chessRL import make_matrix
from chessRL_analytics import calculate_material_adv
import copy

env1 = gym.make('Chess-v0')
state = env1.reset()

string = "e4 c5 Nc3 Nc6 Qf3 e5"
string = "e4 c5"
game = string.split(' ')
for move in game:
    #move = state.san(action)
    action = state.push_san(move)
                
    #take action
    state,reward,done,_ = env1.step(action)

initial_state = state
initial_env = env1
##print(env1.render())
##print(state.status())

####





class MonteCarloTreeSearchNode():
    def __init__(self, state,env, parent=None, parent_action=None, filter_prop=None, filter_moves=None):
        self.state = copy.deepcopy(state)
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.parent_action = parent_action
        self.filter_prop = filter_prop
        self.material_adv = 0
        self.children = []
        self._number_of_visits = 0
        self._results = defaultdict(int)
        self._results[1] = 0
        self._results[0] = 0
        self._results[-1] = 0
        self._untried_actions = None
        self._untried_actions = self.untried_actions(filter_moves)
        self.steps = []
        return

    def get_legal_actions(self, state, env): 
        '''
        Constructs a list of all
        possible actions from current state.
        Returns a list.
        '''
        legal_moves = [state.san(x) for x in env.legal_moves]
        return legal_moves

    def is_game_over(self):
        '''
        It is the game over condition
        and depends on your game. Returns
        true or false
        '''
        #add a option to stop at a mate line after 
        return state.is_game_over()

    def game_result(self, state):
        '''
        Returns 1 or 0 or -1 depending
        on your state corresponding to win,
        tie or a loss.
        '''
        #if state.is_checkmate():
            #print('CHECKMATE')
            
        #print(state.Outcome().winner)
        #print(state.Outcome.termination().winner)
        #result = state.Outcome.result()
        #print(result)
        #print(state.is_variant_win())
        #print(state.is_variant_draw())
        if state.result() == '1-0':
            #print("WIN")
            return 1
        elif state.result() == '0-1':
            #print("LOST")
            return -1
        elif state.result() == '1/2-1/2':
            #print("DRAW")
            return 0
        else:
            #print("Undefined")
            return 0

    def move(self,action, state, env):
        '''
        Changes the state of your 
        board with a new value. Returns 
        the new state after making a move.
        '''
        #move = self.state.push_san(action)
        move_ = state.push_san(action)
                
        #take action
        state,reward,done,_ = env.step(move_)

        return state, env
        

    def untried_actions(self, filter_moves=None):
        #self._untried_actions = self.state.get_legal_actions() #
        self._untried_actions = self.get_legal_actions(self.state, self.env) #
        if self.filter_prop is not None:
            self._untried_actions = self.filter_actions(filter_moves)

        return self._untried_actions

    def filter_actions(self,filter_moves):
        if self.filter_prop == 'next_piece':
            new_moves = []
            for fil in filter_moves:
                for x in self._untried_actions:
                    if fil == 'p':
                        if not x.isupper():
                            new_moves.append(x)
                    else:
                        if x[0] == fil:
                            new_moves.append(x)
                
            self._untried_actions = new_moves
            
        return self._untried_actions
        

    def q(self):
        wins = self._results[1]
        draws = self._results[0]
        loses = self._results[-1]
        return wins - loses
    
    def n(self):
        return self._number_of_visits

    def expand(self):
        action = self._untried_actions.pop()
        ##print(action)
        #next_state = self.state.move(action) #
        a = copy.deepcopy(self.state)
        b = copy.deepcopy(self.env)
        
        next_state, next_env = self.move(action,a,b) #
        ##print("#########################################")
        #print(self.env.render())
        #print(' ')
        #print(b.render())
        #print(' ')
        #print(next_env.render())
        #print(' ')
        child_node = MonteCarloTreeSearchNode(
                    next_state, parent=self, parent_action=action, env=next_env)

        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        return self.state.is_game_over()

    def rollout(self):
        current_rollout_state = copy.deepcopy(self.state)
        current_rollout_env = copy.deepcopy(self.env)
        #current_rollout_state = self.state
        #current_rollout_env = self.env
        material_adv_list = []
        
        while not current_rollout_state.is_game_over():
            #possible_moves = current_rollout_state.get_legal_actions()
            possible_moves = self.get_legal_actions(current_rollout_state,current_rollout_env)
            
            action = self.rollout_policy(possible_moves)
            #current_rollout_state = current_rollout_state.move(action)
            self.steps.append(action)
            
            current_rollout_state,current_rollout_env  = self.move(action,current_rollout_state,current_rollout_env)
            material_adv_list.append(calculate_material_adv(current_rollout_state))

        #return current_rollout_state.game_result()
        ##print(current_rollout_env.render())
        #print(self.env.render())
        self.material_adv = np.mean(material_adv_list)
        return self.game_result(current_rollout_state)

    def backpropagate(self, result):
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=0.1, par=None):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        
        if par == 'material_adv':
            mat_adv = self.get_choice_material_adv()
            mat_adv = [ -x if x <0 else x*10 for x in mat_adv]
            score = [a * b for a,b in zip(choices_weights,mat_adv)]
            ans = self.children[np.argmax(score)]
        else:
            ans = self.children[np.argmax(choices_weights)]
            
        return ans
    
    def get_choice_weights(self, c_param=0.1):
        choices_weights = [(c.q() / c.n()) + c_param * np.sqrt((2 * np.log(self.n()) / c.n())) for c in self.children]
        return choices_weights

    def get_choice_material_adv(self):
        material_adv = [c.material_adv for c in self.children]
        return material_adv

    def rollout_policy(self, possible_moves):
        '''
        Chooses the moves to visit
        '''
        
        return possible_moves[np.random.randint(len(possible_moves))]

    def _tree_policy(self):
        current_node = self
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node

    def best_action(self, par=None):
        simulation_no = 200
        
        for i in range(simulation_no):
            ##print(i)
            v = self._tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
            
        return self.best_child(c_param=0., par=par), self.get_choice_weights(c_param=0.), self.get_choice_material_adv()









def calculate_MCTS(state, env,filter_prop = None ,filter_moves = None, par=None):
    #root = MonteCarloTreeSearchNode(state = initial_state, env = initial_env)
    root = MonteCarloTreeSearchNode(state = state, env = env,filter_prop = filter_prop,filter_moves = filter_moves)
    selected_node,weights,material_adv  = root.best_action(par)
    
    return root,selected_node,weights,material_adv

#root,selected_node,weights = main()
#print(weights)
#print([ x.parent_action for x in root.children])
#print([ x._number_of_visits for x in root.children])
#print(selected_node.parent_action)
#print(selected_node.steps)
#print(root.children[0]._number_of_visits)
#print(root.children[0]._results)
#print([ x._results[1] for x in root.children])
