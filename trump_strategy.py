from pathlib import Path
import pickle
from functools import reduce

import jass.game.rule_schieber
import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.const import card_strings
from jass.game.const import trump_strings_short

class LogisticRegressionTrump:  # noqa

    def __init__(self):
        self.card_strings = list(card_strings)
        with open(Path("data") / "logistic_regression.pkl", "rb") as file:
            self._model = pickle.load(file)



    def interaction_data(self, hand: np.array, inter: str):     # noqa
        return [reduce(lambda a, b: a & b, [hand[self.card_strings.index(color + feature)] == 1 for feature in inter]) for color in "DHSC"]



    def __get_cards_in_hand(self, obs: GameObservation):    # noqa
        return [[card == 1 for card in obs.hand] + [obs.forehand != -1] + self.interaction_data(obs.hand, "J9")
                + self.interaction_data(obs.hand, "AKQ")]


    def choose_trump(self, obs: GameObservation) -> int:    # noqa
        [trump] = self._model.predict(self.__get_cards_in_hand(obs))
        return trump_strings_short.index(trump[0])
