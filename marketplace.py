from faker import Faker
from abc import ABC, abstractmethod
import random
import numpy as np

random.seed(1)


DEBUG = False
fake = Faker()
fake.seed_instance(0)

places = [fake.city() for _ in range(200)]
names = [fake.name() for _ in range(200)]


class Entity(ABC):
    def __init__(self, name, min_acceptance_thresh):
        self.name = name
        self.current_match = None
        self.preferences = None
        self.min_acceptance_thresh = min_acceptance_thresh

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
        if self.current_match is None and self.p_index < self.min_acceptance_thresh:
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

        if self.get_preference(employer) > self.min_acceptance_thresh:
            return False, None

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
    def __init__(
        self,
        market_size,
        employer_min_acceptance_thresh=None,
        applicant_min_acceptance_thresh=None,
    ):
        self.market_size = market_size
        self.employer_min_acceptance_thresh = (
            employer_min_acceptance_thresh
            if employer_min_acceptance_thresh
            else market_size
        )
        self.applicant_min_acceptance_thresh = (
            applicant_min_acceptance_thresh
            if applicant_min_acceptance_thresh
            else market_size
        )

        self.employers = [
            Employer(e, self.employer_min_acceptance_thresh)
            for e in places[:market_size]
        ]
        self.applicants = [
            Applicant(a, self.applicant_min_acceptance_thresh)
            for a in names[:market_size]
        ]

        print(f"Running marketplace of size {self.market_size}")
        print(
            f"Employers will offer spots to their top {self.employer_min_acceptance_thresh}"
        )
        print(
            f"Applicants will accept spots from their top {self.applicant_min_acceptance_thresh}"
        )

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
        # while any(e.current_match is None for e in self.employers):
        for i in range(self.market_size):
            for e in self.employers:
                offer = e.make_offer()
                if offer:
                    decision, displaced = offer.respond_to_offer(e)
                    if decision:
                        e.current_match = offer
                        if displaced:
                            print(f"{displaced.name} lost their match")
                            displaced.current_match = None

        self.stable_marriage_remaining()

    def stable_marriage_remaining(self):
        # runs stable marriage on anyone not getting a match
        e_unmatched, a_unmatched = self.get_unmatched()
        e_names, a_names = set([e.name for e in e_unmatched]), set(
            [a.name for a in a_unmatched]
        )

        e_prefs, a_prefs = {}, {}
        for e in e_unmatched:
            e_prefs[e.name] = [p.name for p in e.preferences if p.name in a_names]

        for a in a_unmatched:
            a_prefs[a.name] = [p.name for p in a.preferences if p.name in e_names]

        from matching.games import StableMarriage

        game = StableMarriage.create_from_dictionaries(e_prefs, a_prefs)
        ans = game.solve()
        print("Matching residuals")
        print(ans)

        for e_name, a_name in ans.items():
            e = next((e for e in e_unmatched if e.name == e_name.name), None)
            a = next((a for a in a_unmatched if a.name == a_name.name), None)
            e.current_match = a
            a.current_match = e

    def get_unmatched(self):
        return (
            [e for e in self.employers if e.current_match is None],
            [a for a in self.applicants if a.current_match is None],
        )

    def stable_marriage(self):
        from matching.games import StableMarriage

        e_prefs, a_prefs = self.build_dict_prefs()
        game = StableMarriage.create_from_dictionaries(e_prefs, a_prefs)
        ans = game.solve()

        e_res, a_res = [], []

        for e, a in ans.items():
            print(f"{e} -> {a}")
            # print(e_prefs.keys(), repr(e), e in e_prefs, e_prefs["New York"])
            e_res.append(e_prefs[e.name].index(a.name) + 1)
            a_res.append(a_prefs[a.name].index(e.name) + 1)

        e_avg, a_avg = np.mean(e_res), np.mean(a_res)
        print("employer average rank: ", e_avg)
        print("applicants average rank: ", a_avg)
        return e_avg, a_avg

    def print_matching(self):
        for e in self.employers:
            print(f"{e.name} -> {e.current_match.name}")

    ### Statistics
    def compute_avg(self):
        e_avg = np.mean([e.get_cur_match_pref() + 1 for e in self.employers])
        a_avg = np.mean([a.get_cur_match_pref() + 1 for a in self.applicants])
        print("employer average rank: ", e_avg)
        print("applicants average rank: ", a_avg)
        return e_avg, a_avg

    def run_marketplace(self):
        self.build_preferences()
        self.match()


def simulation():
    m = Marketplace(10, applicant_min_acceptance_thresh=4)
    m.build_preferences()
    m.match()
    m.print_matching()
    m.compute_avg()
    m.stable_marriage()


simulation()
