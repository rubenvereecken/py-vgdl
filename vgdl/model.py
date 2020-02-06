# from itertools import combinations_with_replacement as comb_rep_iter
from itertools import product
from scipy.misc import comb
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Union, List, Callable, Any, Dict

from vgdl.core import BasicGame, ACTION, GameState, Action
from vgdl.state import Observation

Statelike = Union[GameState, Observation]

class GraphBuilder:
    def __init__(self, game):
        self.game = game
        actiondict = self.game.get_possible_actions(include_noop=False) # type: Dict
        self.actions = list(actiondict.values())

    def grow_state_graph_bfs(self):
        def _observe(state):
            return state
        return self.grow_graph_bfs(_observe)

    def grow_observation_graph_bfs(self, observer):
        """
        Having a separate method for building observation graphs is useful
        for when the game-state space is too large.
        """
        assert observer.game == self.game
        def _observe(state):
            # assert game.get_game_state() == state
            obs = observer.get_observation()
            return obs
        return self.grow_graph_bfs(_observe)

    def grow_graph_bfs(self, observe: Callable[[GameState], Any]):
        assert not self.game.is_stochastic, 'Deterministic only'

        self.game.reset()
        init_state = self.game.get_game_state()
        init_obs = observe(init_state)
        visited = set([init_obs])
        state_fringe = [init_state]
        obs_fringe = [init_obs]

        graph = OrderedDict()
        # Assumes r(s,a,s') is really just r(s')
        reward_graph = {}

        while obs_fringe:
            current_state = state_fringe.pop(0)
            current_obs = obs_fringe.pop(0)
            visited.add(current_obs)
            graph[current_obs] = OrderedDict()

            if current_state.ended():
                # For completeness, add self-edges for absorbing states
                # NOTE Careful, self edges mess up model-based solvers
                for action in self.actions:
                    # self.graph[current][action] = current
                    graph[current_obs][action] = current_obs
                continue

            for action in self.actions:
                self.game.set_game_state(current_state)

                assert self.game.get_game_state() == current_state
                #     import ipdb; ipdb.set_trace()
                #     raise Exception
                self.game.tick(action)

                neighbor_state = self.game.get_game_state()
                neighbor_obs = observe(neighbor_state)
                reward = neighbor_state.get_reward()

                # Expect deterministic MDPs for now
                assert action not in graph[current_obs] or graph[current_obs][action] == neighbor_obs
                graph[current_obs][action] = neighbor_obs

                # Just validating the r(s,a,s')=r(s') assumption
                assert neighbor_obs not in reward_graph or reward_graph[neighbor_obs] == reward
                reward_graph[neighbor_obs] = reward

                if neighbor_obs not in visited and neighbor_obs not in obs_fringe:
                    obs_fringe.append(neighbor_obs)
                    state_fringe.append(neighbor_state)

        # if init_obs not in reward_graph:
        #     reward_graph[init_obs] = 0

        wrapped = DeterministicGraph(self.game, graph, reward_graph)
        return wrapped

class MDPGraph:
    pass

class DeterministicGraph(MDPGraph):
    def __init__(self, game, graph: Dict[Statelike, Dict[Action, Statelike]],
                 reward_graph: Dict[Statelike, float]):
        self.game = game
        self.graph = graph # type: Dict[Statelike Dict[Action, Statelike]]
        self.reward_graph = reward_graph

        actiondict = self.game.get_possible_actions(include_noop=False) # type: Dict
        self.actions = list(actiondict.values())

        # self.transitions = self._construct_transition_matrix(graph)

    def _construct_transition_matrix(self, graph):
        num_states = len(graph)
        num_actions = len(self.actions)
        transitions = np.empty((num_states, num_actions), dtype=np.uint16)
        assert self.num_states < 2**16, "That's not going to fit mate"

        state_to_idx = { state: state_i for state_i, state in enumerate(graph.keys()) }
        action_to_idx = { action: action_i for action_i, action in enumerate(self.actions) }

        for u, edges in self.graph.items():
            for a, v in edges.items():
                # TODO broken right now, because v can be None
                transitions[state_to_idx[u], action_to_idx[a]] = state_to_idx[v]

        return transitions

    def as_transition_matrix(self) -> 'np.ndarray[S,A,S]':
        """ T holds probabilities for (s, a, s')"""
        T = np.zeros((self.num_states, self.num_actions, self.num_states), dtype=np.float)

        state_to_idx = { state: state_i for state_i, state in enumerate(self.graph.keys()) }
        action_to_idx = { action: action_i for action_i, action in enumerate(self.actions) }

        for u, edges in self.graph.items():
            for a, v in edges.items():
                if v is not None:
                    T[state_to_idx[u], action_to_idx[a], state_to_idx[v]] = 1
                # if v is None, just leave it all 0

        return T

    def as_deterministic_transition_matrix(self) -> 'np.ndarray[S,S]':
        """ T holds resulting state index for (s, a) """
        T = np.zeros((self.num_states, self.num_actions), dtype=int)

        state_to_idx = { state: state_i for state_i, state in enumerate(self.graph.keys()) }
        action_to_idx = { action: action_i for action_i, action in enumerate(self.actions) }

        for u, edges in self.graph.items():
            for a, v in edges.items():
                if v is not None:
                    T[state_to_idx[u], action_to_idx[a]] = state_to_idx[v]
                # if v is None, just leave it all 0

        return T

    def state_to_idx(self) -> Dict[Statelike, int]:
        state_to_idx = { state: state_i for state_i, state in enumerate(self.graph.keys()) }
        return state_to_idx

    def as_reward_vector(self) -> 'np.ndarray[S]':
        rewards = np.fromiter((self.reward_graph[s] for s in self.states()), dtype=np.float)
        return rewards

    def as_networkx_graph(self):
        import networkx as nx
        graph = nx.OrderedMultiDiGraph()
        # Not sure if we could just insert edges and have the nodes be inserted
        # automatically, not sure if order is properly preserved then
        graph.add_nodes_from(self.graph.keys())
        graph.add_edges_from(((u, v, { 'action': a }) \
                              for u, edges in self.graph.items() \
                              for a, v in edges.items()))
        return graph

    def as_indexed_networkx_graph(self):
        import networkx as nx
        graph = nx.MultiDiGraph()
        graph.add_nodes_from(range(self.num_states))
        graph.add_edges_from(((u, self.transitions[u, a], { 'action': a }) \
                              for u, a in product(range(self.num_states), range(self.num_actions))))
        return graph

    def states(self):
        for state in self.graph.keys():
            yield state

    @property
    def num_states(self):
        return len(self.graph)

    @property
    def num_actions(self):
        return len(self.actions)

    def state_any(self, predicates: Union[List[Callable], Callable]):
        if not isinstance(predicates, list):
            predicates = [predicates]

        checks = [False] * len(predicates)

        for state in self.graph.nodes:
            for pred_i, predicate in enumerate(predicates):
                if not checks[pred_i]:
                    checks[pred_i] = predicate(state)
            if all(checks):
                return True

        return False


    def state_all(self, predicates: Union[List[Callable], Callable]):
        if not isinstance(predicates, list):
            predicates = [predicates]

        return all(all(p(s) for p in predicates) for s in self.graph.nodes)
