DEBUG = True
names = [
    "John",
    "Emma",
    "Michael",
    "Sophia",
    "William",
    "Olivia",
    "James",
    "Ava",
    "Alexander",
    "Isabella",
]

places = [
    "New York",
    "London",
    "Paris",
    "Tokyo",
    "Los Angeles",
    "Sydney",
    "Rome",
    "Berlin",
    "Dubai",
    "Moscow",
]
from abc import ABC, abstractmethod
import random

random.seed(1)


class Entity(ABC):
    def __init__(self, name):
        self.name = name
        self.current_match = None
        self.preferences = None

    def __str__(self):
        ans = self.name

        if self.preferences:
            ans += "\nPreferences:"
            for i, p in enumerate(self.preferences):
                ans += f"\n    {i}: {p.name}"
        return ans

    def __repr__(self):
        return str(self)

    def make_preferences(self, options):
        # takes in a list of options and ranks them
        self.preferences = list(random.sample(options, len(options)))
        self.p_index = 0

    def get_preference(self, option):
        return self.preferences.index(option)

    def get_cur_match_pref(self):
        return self.preferences.index(self.current_match)


class Employer(Entity):
    def make_offer(self):
        # makes an offer to a candidate
        if self.current_match is None:
            self.p_index += 1
            if DEBUG:
                print(
                    f"{self.name} making offer to {self.preferences[self.p_index - 1].name}"
                )
            return self.preferences[self.p_index - 1]
        else:
            return None


class Applicant(Entity):
    def respond_to_offer(self, employer):
        # Receives an offer in the form of an employer
        # returns bool, employer
        # bool indicates whether match is accepted, employer indicates if someone was dropped from a match
        if self.current_match is None:
            if DEBUG:
                print(f"{self.name} has no match and accepts {employer.name}")
            self.current_match = employer
            return True, None
        elif self.get_preference(self.current_match) > self.get_preference(employer):
            # the match is better than the current one
            if DEBUG:
                print(
                    f"{self.name} found a better match, accepts {employer.name} ({self.get_preference(employer)}), and discards {self.current_match.name} ({self.get_preference(self.current_match)})"
                )
            last_match = self.current_match
            self.current_match = employer
            return True, last_match
        else:
            print("reject")
            return False, None


class Marketplace(object):
    def __init__(self, market_size):
        self.market_size = market_size
        self.employers = [Employer(e) for e in places[:market_size]]
        self.applicants = [Applicant(a) for a in names[:market_size]]

    def build_dict_prefs(self):
        e_prefs = {e.name: [a.name for a in e.preferences] for e in self.employers}
        a_prefs = {a.name: [e.name for e in a.preferences] for a in self.applicants}
        return e_prefs, a_prefs

    def __str__(self):
        ans = "Employers:\n"
        for e in self.employers:
            ans += f"{e}\n"

        ans += "\nApplicants:\n"
        for a in self.applicants:
            ans += f"{a}\n"

        return ans

    def build_preferences(self):
        # loads preferences for all market participants
        for e in self.employers:
            e.make_preferences(self.applicants)

        for a in self.applicants:
            a.make_preferences(self.employers)

    def match(self):
        # runs match
        while any(e.current_match is None for e in self.employers):
            for e in self.employers:
                offer = e.make_offer()
                if offer:
                    decision, displaced = offer.respond_to_offer(e)
                    if decision:
                        e.current_match = offer
                        if displaced:
                            print(f"{displaced.name} lost their match")
                            displaced.current_match = None

    def alg(self):
        from matching.games import StableMarriage

        game = StableMarriage.create_from_dictionaries(*self.build_dict_prefs())
        ans = game.solve()
        for e, a in ans.items():
            print(f"{e} -> {a}")

    def print_matching(self):
        for e in self.employers:
            print(f"{e.name} -> {e.current_match.name}")

    ### Statistics
    def compute_avg(self):
        import numpy as np

        e_avg = np.mean([e.get_cur_match_pref() + 1 for e in self.employers])
        a_avg = np.mean([a.get_cur_match_pref() + 1 for a in self.applicants])
        print("employer average rank: ", e_avg)
        print("applicants average rank: ", a_avg)


m = Marketplace(10)
m.build_preferences()
print(m)
m.match()
m.print_matching()
m.compute_avg()
print(m.alg())
