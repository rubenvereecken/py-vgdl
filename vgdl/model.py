# from itertools import combinations_with_replacement as comb_rep_iter
from itertools import product
from scipy.misc import comb
import numpy as np
from collections import defaultdict, OrderedDict
from typing import Union, List, Callable

from vgdl.core import BasicGame, ACTION


class StateActionGraph:
    def __init__(self, game):
        self.game = game
        self.graph = None # type: Dict[GameState, Dict[Action, GameState]]

        actiondict = self.game.get_possible_actions() # type: Dict
        self.actions = list(actiondict.values())
        self.actions.remove(ACTION.NOOP)

        # Assumes a single starting state
        self.init_state = self.game.init_state

    def __getstate__(self):
        return { k: v for k, v in self.__dict__.items() if k != 'game' }

    def grow_graph_bfs(self):
        assert not self.game.is_stochastic, 'Deterministic only'

        self.game.reset()
        init_state = self.game.get_game_state()
        visited = set([init_state])
        fringe = [init_state]
        # self.graph.add_node(state)
        self.graph = OrderedDict()

        while fringe:
            current = fringe.pop(0)
            visited.add(current)

            self.graph[current] = OrderedDict()

            if current.ended():
                # for completeness, add self-edges for absorbing states
                for action in self.actions:
                    self.graph[current][action] = current
                    # self.graph.add_edge(current, current, action=action)
                continue

            for action in self.actions:
                self.game.set_game_state(current)

                if self.game.get_game_state() != current:
                    import ipdb; ipdb.set_trace()
                    raise Exception

                self.game.tick(action)

                neighbor = self.game.get_game_state()
                self.graph[current][action] = neighbor
                # self.graph.add_edge(current, neighbor, action=action)

                if neighbor not in visited and neighbor not in fringe:
                    fringe.append(neighbor)

        self.transitions = np.empty((self.num_states, self.num_actions), dtype=np.uint16)
        assert self.num_states < 2**16, 'Thats not going to fit mate'

        state_to_idx = { state: state_i for state_i, state in enumerate(self.states()) }
        action_to_idx = { action: action_i for action_i, action in enumerate(self.actions) }

        for u, edges in self.graph.items():
            for a, v in edges.items():
                self.transitions[state_to_idx[u], action_to_idx[a]] = state_to_idx[v]

        return self.graph

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


    def observations(self, observer_cls: type):
        observer = observer_cls(self.game)
        initial_state = game.

        for state in self.states():


    @classmethod
    def construct(cls, game: BasicGame):
        graph = StateActionGraph(game)
        graph.grow_graph_bfs()
        return graph


# class GraphInspector:
#     """
#     Utilities for running tests on a state action graph
#     """
#     def __init__(g: StateActionGraph):

