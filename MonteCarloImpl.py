import numpy as np
import datetime
import math

from jass.game.const import same_team, next_player
from jass.game.game_observation import GameObservation
from jass.game.rule_schieber import RuleSchieber


class MonteCarloTreeSearch:

    def __init__(self, timeout, c):
        self._timeout = timeout
        self._c = c
        self._rule = RuleSchieber()
    @staticmethod
    def remaining_cards(obs: GameObservation):
        remaining = np.ones(shape=36, dtype=int) - obs.hand
        remaining = np.ones(shape=36, dtype=int) - obs.hand
        # finds indices of cards that have been played and sets them to 0 -> to check if a particular card has been played before
        non_zero_indices = [obs.tricks[np.where(obs.tricks >= 0)]]
        for index in non_zero_indices:
            remaining[index] = 0
        return np.flatnonzero(remaining)

    def montecarlosearch(self, obs: GameObservation) -> int:
        end = datetime.timedelta(seconds=self._timeout)
        start = datetime.datetime.utcnow()

        current_hand = np.flatnonzero(self._rule.get_valid_cards_from_obs(obs))
        if len(current_hand) == 1:  # speedup
            return current_hand[0]
        remaining_cards = self.remaining_cards(obs)

        nr_tries = len(current_hand)
        first_play_tries, first_play_scores, first_play_ucbs = self.play(current_hand, remaining_cards, obs, nr_tries)

        self.simulate(start, end, first_play_tries, first_play_ucbs, first_play_scores, current_hand, remaining_cards, obs, nr_tries)

        return current_hand[first_play_tries.index(max(first_play_tries))]

    def simulate(self,start,  end, tries, ucbs, scores, current_hand, remaining_cards, obs, nr_tries):
        while datetime.datetime.utcnow() - start < end:
            i = ucbs.index(max(ucbs))
            nr_tries = nr_tries + 1
            tries[i] = tries[i] + 1
            scores[i] = scores[i] + self.expand(
                current_hand[i], remaining_cards.tolist(), obs)
            ucbs[i] = self.ucb(
                scores[i], tries[i], nr_tries)

    @staticmethod
    def play_card(obs: GameObservation, card_to_play):
        obs.hand[card_to_play] = 0

    def play(self, hand, remaining, obs, nr_tries):
        tries = []
        for _ in hand:
            tries.append(1)
        score = []
        for s in hand:
            score.append(self.expand(s, remaining.tolist(), obs))
        ucb1 = []
        for s, t in zip(score, tries):
            ucb1.append(self.ucb(s, t, nr_tries))

        return tries, score, ucb1

    def ucb(self, score, total, tries):
        return score / tries + self._c * math.sqrt(math.log(total) / tries)

    def expand(self, played, remaining, obs):
        result = 0
        np.random.shuffle(remaining)
        obs = self.copy_observation(obs)
        me = obs.player

        self.play_card(obs, played)
        self.update_observation(obs, played)
        while obs.nr_tricks < 9:
            is_last = obs.nr_tricks == 8
            while obs.nr_cards_in_trick < 4:
                if obs.player == me:
                    cards = self._rule.get_valid_cards_from_obs(obs)
                    card_played = np.random.choice(np.flatnonzero(cards))
                    self.play_card(obs, card_played)
                else:
                    # don't know what others have so choose at random
                    card_played = remaining.pop()
                self.update_observation(obs, card_played)
            obs.player = self._rule.calc_winner(
                obs.current_trick, obs.trick_first_player[obs.nr_tricks], obs.trump)
            if same_team[me][obs.player]:
                result = result + self._rule.calc_points(
                    obs.current_trick, is_last, obs.trump)

            obs.current_trick = np.full(shape=4, fill_value=-1, dtype=np.int32)
            obs.nr_tricks += 1
            obs.nr_cards_in_trick = 0
            if not is_last:
                obs.trick_first_player[obs.nr_tricks] = obs.player

        return result

    @staticmethod
    def copy_observation(to_copy: GameObservation):
        res = GameObservation()
        res.player = to_copy.player
        res.hand = np.copy(to_copy.hand)
        res.trump = to_copy.trump
        res.current_trick = np.copy(to_copy.current_trick)
        res.trick_first_player = np.copy(to_copy.trick_first_player)
        res.nr_cards_in_trick = to_copy.nr_cards_in_trick
        res.nr_tricks = to_copy.nr_tricks
        return res

    @staticmethod
    def update_observation(obs: GameObservation, card_played: int):
        obs.current_trick[obs.nr_cards_in_trick] = card_played
        obs.nr_cards_in_trick = obs.nr_cards_in_trick + 1
        obs.player = next_player[obs.player]