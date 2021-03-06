from pathlib import Path
import pickle
from functools import reduce

import jass.game.rule_schieber
import numpy as np
from jass.game.game_observation import GameObservation
from jass.game.const import card_strings
from tensorflow import keras

from jass.game.const import (PUSH,  trump_ints,
                             trump_strings_short)


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

class DeepNeuralTrump: # noqa

    def __init__(self):
        self.int_to_trump = trump_ints + [PUSH] * 5
        self._model = keras.models.load_model(Path("data") / "trump_dnn.h5")


    def choose_trump(self, obs: GameObservation) -> int: # noqa
        prediction = np.argsort(self._model.predict([
            list(obs.hand.astype(float)) +
            [float(obs.forehand + 1)]
        ]))[0][::-1]

        result = [self.int_to_trump[trump] for trump in prediction]
        return next(trump for trump in result if obs.forehand == -1 or trump != PUSH)

