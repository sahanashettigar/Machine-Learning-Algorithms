import numpy as np


class HMM:
    """
    HMM model class
    Args:
        A: State transition matrix
        states: list of states
        emissions: list of observations
        B: Emmision probabilites
    """

    def __init__(self, A, states, emissions, pi, B):
        self.A = A
        self.B = B
        self.states = states
        self.emissions = emissions
        self.pi = pi
        self.N = len(states)
        self.M = len(emissions)
        self.make_states_dict()

    def make_states_dict(self):
        """
        Make dictionary mapping between states and indexes
        """
        self.states_dict = dict(zip(self.states, list(range(self.N))))
        self.emissions_dict = dict(
            zip(self.emissions, list(range(self.M))))

    def viterbi_algorithm(self, seq):
        """
        Function implementing the Viterbi algorithm
        Args:
            seq: Observation sequence (list of observations. must be in the emmissions dict)
        Returns:
            nu: Porbability of the hidden state at time t given an obeservation sequence
            hidden_states_sequence: Most likely state sequence
        """
        if(len(seq) == 0):
            return []
        self.make_states_dict()
        emission_in_num = []
        for item in seq:
            emission_in_num.append(self.emissions_dict[item])
        probabilites = []
        probabilites.append(
            [self.pi[state]*self.B[state][emission_in_num[0]]
             for state in range(len(self.states))]
        )
        for i in range(1, len(emission_in_num)):
            prev_node_prob = probabilites[len(probabilites)-1]
            curr_node_prob = []
            for col in range(len(self.A[0, :])):
                temparr = []
                for state in range(len(self.states)):
                    temparr.append(prev_node_prob[state]*self.A[state]
                                   [col]*self.B[col][emission_in_num[i]])
                curr_node_prob.append(max(temparr))
            probabilites.append(curr_node_prob)
        hidden_states_seq_op = []
        for i in probabilites:
            best_hidden_state = self.states[np.argmax(i)]
            hidden_states_seq_op.append(best_hidden_state)
        return hidden_states_seq_op
