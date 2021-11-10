import math

from jass.agents.agent import Agent
from jass.game.rule_schieber import RuleSchieber
from jass.game.game_util import *
from trump_strategy import *
from play_strategy import MCTSPlayStrategy

class PlayerLogisticRegrMCTS(Agent):

    def __init__(self):
        self.trump_strat = LogisticRegressionTrump()
        self.play_strat = MCTSPlayStrategy(0.2, 1)
        self._rule = RuleSchieber()
        self._rng = np.random.default_rng()

    def action_trump(self, obs: GameObservation) -> int:
        return self.trump_strat.choose_trump(obs)

    def action_play_card(self, obs: GameObservation) -> int:
        # valid_cards = self._rule.get_valid_cards_from_obs(obs)
        # card = self._rng.choice(np.flatnonzero(valid_cards))
        # #self._logger.debug('Played card: {}'.format(card_strings[card]))
        # return card
        return self.play_strat.choose_card(obs)
