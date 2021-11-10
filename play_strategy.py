import datetime
import math
import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber
from jass.game.const import *
from MonteCarloImpl import MonteCarloTreeSearch
class MCTSPlayStrategy:

    def __init__(self, timeout, c):
        self._timeout = timeout
        self._c = c
        self._rule = RuleSchieber()
        self._strategy = MonteCarloTreeSearch(timeout, c)

    def choose_card(self, obs: GameObservation):
        return self._strategy.montecarlosearch(obs)
