from abc import ABC, abstractmethod
import random

from tqdm import tqdm
import numpy as np
from faker import Faker
from matching.games import StableMarriage
import test_config

DEBUG = False
RANDOM = True


def handle_offer(
    employer,
    employer_sincerity,
    applicant,
    applicant_decision,
    applicant_sincerity,
):
    # Four cases:
    # 1. Sincere offer, sincere acceptance: work as normal
    # 2. Sincere offer, insincere acceptance: employer thinks it's a match, but it only is
    #    binding if it is preferable for the applicant. Applicant doesn't inform other matches of broken match
    # 3. Insincere offer, sincere acceptance: Applicant thinks its a match, but it only is binding
    #    if it is preferred by the employer. Employer doesn't inform other matches of broken match
    # 4. Insincere offer, insincere acceptance: Only binding for each if it's preferred...
    #    neither informs broken matches

    if applicant_decision is False:
        return

    if employer_sincerity and applicant_sincerity:
        a_last_match = applicant.current_match
        e_last_match = employer.current_match
        employer.current_match = applicant
        applicant.current_match = employer

        # inform old matches
        if a_last_match is not None:
            a_last_match.current_match = None
        if e_last_match is not None:
            e_last_match.current_match = None
    elif not employer_sincerity and applicant_sincerity:
        # employer is a liar, applicant isn't
        last_match = applicant.current_match
        applicant.current_match = employer
        if last_match is not None:
            # inform old match
            last_match.current_match = None
        if employer.current_match is None:
            # employer only makes this his match if he has no current match
            # if he has a match, its by definition a higher preference
            employer.current_match = applicant
    elif employer_sincerity and not applicant_sincerity:
        employer.current_match = applicant
        if applicant.current_match is None or applicant.get_preference(
            employer
        ) < applicant.get_preference(applicant.current_match):
            applicant.current_match = employer
            # don't inform broken matches
    else:
        # both are lying
        if employer.current_match is None:
            # employer only makes this his match if he has no current match
            # if he has a match, its by definition a higher preference
            employer.current_match = applicant
        if applicant.current_match is None or applicant.get_preference(
            employer
        ) < applicant.get_preference(applicant.current_match):
            applicant.current_match = employer
            # don't inform broken matches


class Entity(ABC):
    def __init__(
        self,
        name,
        desirability,
        min_acceptance_thresh,
        preference_distribution,
        rule_follower=False,
        liar=False,
        noise=20,
    ):
        self.name = name
        self.desirability = desirability
        self.current_match = None
        self.preferences = None
        self.preference_distribution = preference_distribution
        self.min_acceptance_thresh = min_acceptance_thresh
        self.liar = liar
        self.rule_follower = rule_follower

        if self.liar and self.rule_follower:
            # Can't be both a liar and rule follower, but could happen based on stochasticity
            # In case of conflict, we'll make them a liar
            self.rule_follower = False

        self.noise = noise

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
        # takes in a list of options and ranks them according to distribution,
        self.preferences = [o for o in options]
        if self.preference_distribution == "Normal":
            self.preferences.sort(
                key=lambda x: x.desirability + random.gauss(0, self.noise)
            )
        elif self.preference_distribution == "Uniform":
            random.shuffle(self.preferences)
        else:
            raise Exception(
                f"Could not handle preference_distribution: {self.preference_distribution}"
            )
        self.p_index = 0

    def get_preference(self, option):
        return self.preferences.index(option)

    def get_cur_match_pref(self):
        return self.preferences.index(self.current_match)


class Employer(Entity):
    def make_offer(self):
        # Returns Option[Applicant offering a match or None], Sincerity (bool)
        # Sincerity is true if he's not a liar
        if self.rule_follower:
            # never makes an offer if he's a rule follower
            return None, True
        elif self.liar and self.p_index <= self.min_acceptance_thresh:
            # if they're liars and the person is within their acceptance thresh, make an insincere offer
            self.p_index += 1
            return self.preferences[self.p_index - 1], False
        elif self.current_match is None and self.p_index <= self.min_acceptance_thresh:
            # Checks if there is no match and this candidate is above the threshold
            self.p_index += 1
            if DEBUG:
                print(
                    f"{self.name} making offer to {self.preferences[self.p_index - 1].name}"
                )
            return self.preferences[self.p_index - 1], True
        else:
            return None, True


class Applicant(Entity):
    def respond_to_offer(self, employer):
        # Receives an offer in the form of an employer
        # returns acceptance (bool), sincerity (bool)
        # bool indicates whether match is accepted, sincerity indicates whether applicant is a liar

        if not self.liar and (
            self.get_preference(employer) > self.min_acceptance_thresh
            or self.rule_follower
        ):
            # Not at acceptance threshold or is a rule follower
            return False, True

        if self.liar:
            # if they're a liar, they accept
            return True, False
        elif self.current_match is None:
            # No current match, so accept
            if DEBUG:
                print(f"{self.name} has no match and accepts {employer.name}")
            return True, True
        elif self.get_preference(self.current_match) > self.get_preference(employer):
            # the new offer is better than the current match
            if DEBUG:
                print(
                    f"{self.name} found a better match, accepts {employer.name} ({self.get_preference(employer)}), and discards {self.current_match.name} ({self.get_preference(self.current_match)})"
                )
            return True, True
        else:
            # New offer is worse than current match and they're not lying
            return False, True


class Marketplace(object):
    """This object creates a marketplace simulation

    Args:
    market_size (int): how many people are in the market
    employer_min_acceptance_thresh (int): Employers will only offer jobs to applicants ranked at least this highly on their list
    applicant_min_acceptance_thresh (int): Applicants will only accept jobs from employers ranked at least this highly
    employer/applicant_preference_distribution (str): Way employer/applicant makes preferences:
        'Normal': sort desirability scores with added bias (random sample from N(0,30))
        'Uniform': shuffle irrespective of desirability
    employer/applicant_desirability_dist (int, int): parameterizes the normal distribution from which employer/applicant desirabilities are drawn
    employer/applicant_rule_follower_rate (float between 0 and 1): what proportion of employers/applicants follow the rules (e.g. don't pre-match)
    employer/applicant_noise (int): parameterizes the normal distribution N(0, noise) that represents an applicant/employers desire for a particular employer/applicant
    employer/applicant_lie_rate (float between 0 and 1): what proportion of employers/applicants lies. Liars offer/accept matches but they are being unreliable.
    """

    def __init__(
        self,
        market_size=20,
        employer_min_acceptance_thresh=None,
        applicant_min_acceptance_thresh=None,
        employer_preference_distribution: str = "Normal",
        applicant_preference_distribution: str = "Normal",
        employer_desirability_params=(50, 20),
        applicant_desirability_params=(50, 20),
        employer_rule_follower_rate=0,
        applicant_rule_follower_rate=0,
        employer_noise=20,
        applicant_noise=20,
        employer_lie_rate=0,
        applicant_lie_rate=0,
    ):
        params = locals()
        unwanted = ["self"]
        params = {key: value for key, value in params.items() if key not in unwanted}
        self.specification = params

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
        self.employer_preference_distribution = employer_preference_distribution
        self.applicant_preference_distribution = applicant_preference_distribution
        fake = Faker()
        fake.seed_instance(0)
        if not RANDOM:
            np.random.seed(0)
            random.seed(0)

        employers = list(set([fake.city() for _ in range(250)]))
        applicants = list(set([fake.name() for _ in range(250)]))
        if not RANDOM:
            np.random.seed(0)
        employer_desirabilities = np.random.normal(
            *employer_desirability_params, market_size
        )

        if not RANDOM:
            np.random.seed(0)
        applicant_desirabilities = np.random.normal(
            *applicant_desirability_params, market_size
        )

        self.employers = [
            Employer(
                e,
                d,
                self.employer_min_acceptance_thresh,
                self.employer_preference_distribution,
                rule_follower=random.random() < employer_rule_follower_rate,
                liar=random.random() < employer_lie_rate,
                noise=employer_noise,
            )
            for e, d in zip(employers[:market_size], employer_desirabilities)
        ]
        self.applicants = [
            Applicant(
                a,
                d,
                self.applicant_min_acceptance_thresh,
                self.applicant_preference_distribution,
                rule_follower=random.random() < applicant_rule_follower_rate,
                liar=random.random() < applicant_lie_rate,
                noise=applicant_noise,
            )
            for a, d in zip(applicants[:market_size], applicant_desirabilities)
        ]

    @classmethod
    def test_config(with_rule_breakers=False, with_liars=False):
        # removes all randomness and generates a simple 10 person example
        m = Marketplace.__new__(Marketplace)
        m.market_size = test_config.market_size
        m.employers = []
        m.applicants = []
        for name, prefs in zip(test_config.units, test_config.unit_preferences):
            e = Employer(
                name,
                0,
                test_config.min_acceptance_thresh,
                test_config.preference_distribution,
            )
            e.preference_weights = prefs
            e.p_index = 0
            m.employers.append(e)

        for name, prefs in zip(test_config.officers, test_config.officer_preferences):
            a = Applicant(
                name,
                0,
                test_config.min_acceptance_thresh,
                test_config.preference_distribution,
            )
            a.preference_weights = prefs
            a.p_index = 0
            m.applicants.append(a)

        for e in m.employers:
            e.preferences = [
                (a, weight) for (a, weight) in zip(m.applicants, e.preference_weights)
            ]
            e.preferences.sort(key=lambda x: x[1])
            e.preferences = [tup[0] for tup in e.preferences]
        for a in m.applicants:
            a.preferences = [
                (e, weight) for (e, weight) in zip(m.employers, a.preference_weights)
            ]
            a.preferences.sort(key=lambda x: x[1])
            a.preferences = [tup[0] for tup in a.preferences]
        return m

    def description(self):
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
        # runs matching algorithm
        # 1. Do a round for every applicant in the market
        # 2. Each employer makes an offer to his top choice (that they haven't offered to yet)
        #    if they don't have a match
        # 3. Applicants either accept, accept and displace a match, or reject
        # 4. Once all is over, if there are remaining matchless people run Gale-Shapley
        for i in range(self.market_size):
            for e in self.employers:
                offer, offer_sincerity = e.make_offer()
                if offer is None:
                    continue
                decision, applicant_sincerity = offer.respond_to_offer(e)
                if not decision:
                    continue
                handle_offer(e, offer_sincerity, offer, decision, applicant_sincerity)

        self.stable_marriage_remaining()

    def stable_marriage_remaining(self):
        # runs stable marriage on anyone not matched or erroneously in a match
        e_unmatched, a_unmatched = self.get_unmatched()
        e_names, a_names = set([e.name for e in e_unmatched]), set(
            [a.name for a in a_unmatched]
        )

        e_prefs, a_prefs = {}, {}
        for e in e_unmatched:
            e_prefs[e.name] = [p.name for p in e.preferences if p.name in a_names]

        for a in a_unmatched:
            a_prefs[a.name] = [p.name for p in a.preferences if p.name in e_names]

        game = StableMarriage.create_from_dictionaries(e_prefs, a_prefs)
        ans = game.solve()

        for e_name, a_name in ans.items():
            e = next((e for e in e_unmatched if e.name == e_name.name), None)
            a = next((a for a in a_unmatched if a.name == a_name.name), None)
            e.current_match = a
            a.current_match = e

    def get_unmatched(self):
        # check if any matches are not aligned
        for e in self.employers:
            for a in self.applicants:
                if e.current_match == a and a.current_match != e:
                    e.current_match = None
                elif a.current_match == e and e.current_match != a:
                    a.current_match = None

        return (
            [e for e in self.employers if e.current_match is None],
            [a for a in self.applicants if a.current_match is None],
        )

    def stable_marriage(self):
        e_prefs, a_prefs = self.build_dict_prefs()
        game = StableMarriage.create_from_dictionaries(e_prefs, a_prefs)
        ans = game.solve()

        e_res, a_res = [], []

        for e, a in ans.items():
            e_res.append(e_prefs[e.name].index(a.name) + 1)
            a_res.append(a_prefs[a.name].index(e.name) + 1)

        return np.array(e_res), np.array(a_res)

    def print_matching(self):
        for e in self.employers:
            print(
                f"{e.name} ({e.get_cur_match_pref()}) -> {e.current_match.name} ({e.current_match.get_cur_match_pref()})"
            )

    def get_stats(self):
        """Returns a ton of statistics for the simulation"""
        stats = {}
        raw_stats = {}
        for prefix, li in zip(["sim_e_", "sim_a_"], [self.employers, self.applicants]):
            # Stats for all entities
            temp = np.array([e.get_cur_match_pref() + 1 for e in li])
            stats[prefix + "mean"] = np.mean(temp)
            stats[prefix + "std"] = np.std(temp)
            raw_stats[prefix] = temp
            # Stats for only rule followers
            temp = np.array([e.get_cur_match_pref() + 1 for e in li if e.rule_follower])
            if len(temp) > 0:
                stats[prefix + "rulefollower_mean"] = np.mean(temp)
                stats[prefix + "rulefollower_std"] = np.std(temp)
            raw_stats[prefix + "rulefollower"] = temp
            # Stats for only rule breakers
            temp = np.array(
                [e.get_cur_match_pref() + 1 for e in li if not e.rule_follower]
            )
            if len(temp) > 0:
                stats[prefix + "rulebreaker_mean"] = np.mean(temp)
                stats[prefix + "rulebreaker_std"] = np.std(temp)
            raw_stats[prefix + "rulebreaker"] = temp

            temp = np.array([e.get_cur_match_pref() + 1 for e in li if e.liar])
            if len(temp) > 0:
                stats[prefix + "liar_mean"] = np.mean(temp)
                stats[prefix + "liar_std"] = np.std(temp)
            raw_stats[prefix + "liar"] = temp

            temp = np.array([e.get_cur_match_pref() + 1 for e in li if not e.liar])
            if len(temp) > 0:
                stats[prefix + "truther_mean"] = np.mean(temp)
                stats[prefix + "truther_std"] = np.std(temp)
            raw_stats[prefix + "truther"] = temp

        # Stats from stable marriage algorithm
        e_res, a_res = self.stable_marriage()
        stats["alg_e_mean"] = np.mean(e_res)
        stats["alg_e_std"] = np.std(e_res)
        stats["alg_a_mean"] = np.mean(a_res)
        stats["alg_a_std"] = np.std(a_res)
        raw_stats["alg_e"] = np.mean(e_res)
        raw_stats["alg_a"] = np.mean(a_res)
        return stats, raw_stats

    def get_desirability_stats(self):
        """Returns distribution of desirability for each place. A vibe check for simulated data

        1. {employer_name: array of positions employer_name is preferenced by applicants}
        2. {applicant_name: array of positions applicant_name is preferenced by employers}
        """
        employer_desirabilities = {}
        for e in self.employers:
            temp = [
                i + 1
                for a in self.applicants
                for i, ap in enumerate(a.preferences)
                if ap.name == e.name
            ]

            employer_desirabilities[e.name] = temp

        applicant_desirabilities = {}
        for a in self.applicants:
            temp = [
                i + 1
                for e in self.employers
                for i, ep in enumerate(e.preferences)
                if ep.name == a.name
            ]

            applicant_desirabilities[a.name] = temp

        return employer_desirabilities, applicant_desirabilities

    def run_marketplace(self):
        self.build_preferences()
        self.match()
