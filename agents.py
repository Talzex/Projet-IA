import os
import pickle
import math
import time
import numpy as np


class Agent:
    def __init__(self, sim, render=None):
        self.sim = sim
        self.render = render

        self.accuracy = 0.25

        self.actions = []
        self.states = []
        self.final_state = 0

        # Initializing
        print('Scanning states...')
        self.build_states()

        print('Initialize agent...')
        self.initialize()

        self.update_texts()

    def update_texts(self):
        """Update the text in the render"""
        if self.render is not None:
            self.render.text = []
            for state in self.states:
                if state is not self.final_state:
                    value = self.get_value(state)
                    if value is not None:
                        f = max(0., min(1., value/(-8.)))
                        value = '%d' % int(value)
                        color = (int(255*f), 30, int(255*(1-f)))
                        self.render.text.append(
                            [value, self.state_to_position(state), color])

    def position_to_state(self, position):
        """Takes a position in parameter and outputs the tuple representing the state"""
        tmp = np.round((position + self.sim.field_size/2.) / self.accuracy)
        return (int(tmp[0]), int(tmp[1]))

    def state_to_position(self, state):
        """Takes a state and give the corresponding position"""
        l, w = self.sim.field_size
        position = np.array(state)*self.accuracy - self.sim.field_size/2.
        return position

    def build_states(self):
        """Scans all possible states"""
        self.actions = list(range(self.sim.orientations))

        # 0 is a (terminal) success state
        self.states.append(self.final_state)

        l, w = self.sim.field_size
        L = math.ceil(l/self.accuracy)
        W = math.ceil(w/self.accuracy)

        for X in range(L+1):
            for Y in range(W+1):
                self.states.append((X, Y))

    def pick_random_action(self):
        """Agent picks a random action"""
        return np.random.randint(0, len(self.actions))

    # To be implemented by agent

    def initialize(self):
        """Initializes the agent"""
        pass

    def get_value(self, state):
        """Gets the current value estimation V(s) of a state"""
        return None

    def pick_action(self, state):
        """Agent decides which action to take"""
        return self.pick_random_action()

    def learn(self, steps=None):
        """Agent learns"""
        pass


class ValueIterationAgent(Agent):
    def get_value(self, state):
        return self.values[state]

    def initialize(self):
        """Build the (deterministic) transition model"""
        self.model = {}
        self.values = {}
        self.values[self.final_state] = 0

        for state in self.states:
            self.values[state] = 0

        if os.path.exists('model.data'):
            f = open('model.data', 'rb')
            self.model = pickle.load(f)
        else:
            print('No model.data found, generating it...')
            n = 0
            for state in self.states:
                n += 1
                print('%d / %d...' % (n, len(self.states)))
                for action in self.actions:
                    self.sim.ball = self.state_to_position(state)
                    goal = self.sim.kick(action)
                    if goal:
                        self.model[(state, action)] = (-1, self.final_state)
                    else:
                        new_state = self.position_to_state(self.sim.ball)
                        self.model[(state, action)] = (-1, new_state)
            f = open('model.data', 'wb')
            pickle.dump(self.model, f)

    def pick_action(self, state, include_score=False):
        best = 0
        best_action = 0
        for a in self.actions:
            reward, next_state = self.model[(state, a)]
            value = self.values[next_state]
            if next_state == self.final_state:
                return a
            elif(reward + value > best):
                best_action = a
                best = value + reward
        return best_action

    def learn(self, steps=None):
        """
        Doing one step of value iteration, we update value function
        for each state using Bellman equation
        """
        self.sim.iterations += 1
        for s in self.states:
            best_action = self.pick_action(s)
            reward, arrival_state = self.model[(s, best_action)]
            best_value = self.values[arrival_state]
            if (arrival_state == self.final_state):
                 self.values[s] = self.final_state + reward
            else : 
                self.values[s] =  best_value + reward 
        self.update_texts()


class ModelFreeAgent(Agent):
    def default_q_value(self):
        return None

    def initialize(self):
        """
        Initializes count and count_states to 0
        Initializes Q to default q value everywhere
        """
        self.count = {}
        self.count_states = {}
        self.Q = {}

        for state in self.states:
            self.count_states[state] = 0
            for action in self.actions:
                self.Q[(state, action)] = self.default_q_value()
                self.count[(state, action)] = 0

    def pick_action(self, state, explore=False):
        if explore:
            # Exploring using exponential decay
            epsilon = np.exp(-self.count_states[state]/100.0)

            if np.random.rand() < epsilon:
                return self.pick_random_action()

        # Selecting best Q values for this state
        best = None
        for action in self.actions:
            if self.Q[(state, action)] is not None:
                if best is None or self.Q[(state, action)] > best[1]:
                    best = (action, self.Q[(state, action)])

        # If we found no Q value, we select one random action
        if best is None:
            return self.pick_random_action()

        return best[0]

    def get_value(self, state):
        """To compute the value, we simply get V(s) = Q(s, a_max)"""
        best_action = self.pick_action(state, explore=False)

        return self.Q[(state, best_action)]

    def learn(self, steps=None):
        """
        Learning, we do steps runs (5000 if not provided)
        We only keep the 10 last hits, and call update_q function with our observations
        """
        N = steps if steps is not None else 5000

        for run in range(N):
            print('Run %d/%d...' % (run+1, N))
            self.sim.reset_ball()
            self.sim.iterations += 1
            episode = []
            over = False
            while not over and len(episode) < 50:
                state = self.position_to_state(self.sim.ball)
                action = self.pick_action(state, explore=True)
                over = self.sim.kick(action)
                episode.append((state, action))

            if len(episode) < 50:
                episode = episode[-10:]

                n = len(episode)
                final_score = -n
                for k in range(n):
                    state, action = episode[k]
                    self.count_states[state] += 1
                    if k == n-1:
                        next_state = self.final_state
                    else:
                        next_state = episode[k+1][0]

                    self.update_q(state, action, -1,
                                      next_state, final_score)
                    final_score += 1

                self.update_texts()
                self.sim.render(self.render)

    # To be implemented by specific model free agent

    def update_q(self, state, action, reward, next_state, returns):
        """
        Update the score of the model-free agent with an observation
        - state, action: the state and action applied
        - score
        """
        pass


class MonteCarloAgent(ModelFreeAgent):
    def update_q(self, state, action, reward, next_state, returns):
        """
        Do a Monte Carlo update of the Q function (using returns)
        If we have no value for the Q function, we update it to the given returns, else we
        use incremental average to update it
        """
        # XXX: TODO


class QLearningAgent(ModelFreeAgent):
    def default_q_value(self):
        return -10

    def update_q(self, state, action, reward, next_state, returns):
        """
        Do a Q-Learning update of the Q function (reward and next state)
        We first find the next action that would be applied in next state using the current
        policy (except if score is terminal)
        And then update Q using incremental average
        """
        # XXX: TODO


def get_agent():
    """
    Agent to use
    """
    #return Agent
    return ValueIterationAgent
    #return MonteCarloAgent
    #return QLearningAgent
