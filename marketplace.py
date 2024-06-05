from faker import Faker
from tqdm import tqdm
from abc import ABC, abstractmethod
import random
import numpy as np
import pandas as pd
from collections import defaultdict

DEBUG = False
RANDOM = True


class Entity(ABC):
    def __init__(
        self,
        name,
        desirability,
        min_acceptance_thresh,
        preference_distribution,
        rule_follower=False,
        lie_rate=0,
        noise=20,
    ):
        self.name = name
        self.desirability = desirability
        self.current_match = None
        self.preferences = None
        self.preference_distribution = preference_distribution
        self.min_acceptance_thresh = min_acceptance_thresh
        self.lie_rate = lie_rate
        self.rule_follower = rule_follower
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

    def should_lie(self):
        return random.random() < self.lie_rate


class Employer(Entity):
    def make_offer(self):
        # makes an offer to a candidate
        if self.rule_follower:
            return None
        elif self.current_match is None and self.p_index <= self.min_acceptance_thresh:
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

        if self.rule_follower:
            return False, None

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
            return False, None


class Marketplace(object):
    """This object creates a marketplace simulation

    Args:
    market_size (int): how many people are in the market
    employer_min_acceptance_thresh (int): Employers will only offer jobs to applicants ranked at least this highly on their list
    applicant_min_acceptance_thresh (int): Applicants will only accept jobs from employers ranked at least this highly
    employer_preference_distribution (str): Way employer makes preferences:
        'Normal': sort desirability scores with added bias (random sample from N(0,30))
        'Uniform': shuffle irrespective of desirability
    applicant_preference_distribution (str): Way applicant makes preferences:
        'Normal': sort desirability scores with added bias (random sample from N(0,30))
        'Uniform': shuffle irrespective of desirability

    """

    def __init__(
        self,
        market_size=20,
        employer_min_acceptance_thresh=None,
        applicant_min_acceptance_thresh=None,
        employer_preference_distribution: str = "Normal",
        applicant_preference_distribution: str = "Normal",
        place_des_dist=(50, 20),
        app_des_dist=(50, 20),
        employer_rule_follower_rate=0,
        applicant_rule_follower_rate=0,
        employer_noise=20,
        applicant_noise=20,
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

        places = list(set([fake.city() for _ in range(250)]))
        names = list(set([fake.name() for _ in range(250)]))
        if not RANDOM:
            np.random.seed(0)
        place_des = np.random.normal(*place_des_dist, market_size)

        if not RANDOM:
            np.random.seed(0)
        app_des = np.random.normal(*app_des_dist, market_size)

        self.employers = [
            Employer(
                e,
                d,
                self.employer_min_acceptance_thresh,
                self.employer_preference_distribution,
                rule_follower=random.random() < employer_rule_follower_rate,
                noise=employer_noise,
            )
            for e, d in zip(places[:market_size], place_des)
        ]
        self.applicants = [
            Applicant(
                a,
                d,
                self.applicant_min_acceptance_thresh,
                self.applicant_preference_distribution,
                rule_follower=random.random() < applicant_rule_follower_rate,
                noise=applicant_noise,
            )
            for a, d in zip(names[:market_size], app_des)
        ]

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
                            if DEBUG:
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
        from matching.games import StableMarriage

        e_prefs, a_prefs = self.build_dict_prefs()
        game = StableMarriage.create_from_dictionaries(e_prefs, a_prefs)
        ans = game.solve()

        e_res, a_res = [], []

        for e, a in ans.items():
            # print(f"{e} -> {a}")
            e_res.append(e_prefs[e.name].index(a.name) + 1)
            a_res.append(a_prefs[a.name].index(e.name) + 1)

        return np.array(e_res), np.array(a_res)
        # e_avg, a_avg = np.mean(e_res), np.mean(a_res)
        # return e_avg, a_avg

    def print_matching(self):
        for e in self.employers:
            print(f"{e.name} -> {e.current_match.name}")

    ### Statistics
    def get_stats(self):
        """Returns statistics for the simulation

        1. Array of employer received preferences
        2. Array of applicant received preferences
        """
        stats = {}
        for prefix, li in zip(["sim_e_", "sim_a_"], [self.employers, self.applicants]):
            temp = np.array([e.get_cur_match_pref() + 1 for e in li])
            stats[prefix + "mean"] = np.mean(temp)
            stats[prefix + "std"] = np.std(temp)
            temp = np.array([e.get_cur_match_pref() + 1 for e in li if e.rule_follower])
            if len(temp) > 0:
                stats[prefix + "rulefollower_mean"] = np.mean(temp)
                stats[prefix + "rulefollower_std"] = np.std(temp)
            # else:
            #     stats[prefix + "rulefollower_mean"] = np.nan
            #     stats[prefix + "rulefollower_std"] = np.nan

            temp = np.array(
                [e.get_cur_match_pref() + 1 for e in li if not e.rule_follower]
            )
            if len(temp) > 0:
                stats[prefix + "rulebreaker_mean"] = np.mean(temp)
                stats[prefix + "rulebreaker_std"] = np.std(temp)
            # else:
            #     stats[prefix + "rulebreaker_mean"] = np.nan
            #     stats[prefix + "rulebreaker_std"] = np.nan

        e_res, a_res = self.stable_marriage()
        stats["alg_e_mean"] = np.mean(e_res)
        stats["alg_e_std"] = np.std(e_res)
        stats["alg_a_mean"] = np.mean(a_res)
        stats["alg_a_std"] = np.std(a_res)

        return stats

    def get_desirability_stats(self):
        """Returns distribution of desirability

        1. {employer_name: array of positions employer_name is preferenced by applicants}
        2. {applicant_name: array of positions applicant_name is preferenced by employers}
        """
        employer_desirabilities = {}
        for e in self.employers:
            temp = []
            for a in self.applicants:
                for i, ap in enumerate(a.preferences):
                    if ap.name == e.name:
                        temp.append(i + 1)

            employer_desirabilities[e.name] = temp

        applicant_desirabilities = {}
        for a in self.applicants:
            temp = []
            for e in self.employers:
                for i, ep in enumerate(e.preferences):
                    if ep.name == a.name:
                        temp.append(i + 1)

            applicant_desirabilities[a.name] = temp

        return employer_desirabilities, applicant_desirabilities

    def run_marketplace(self):
        self.build_preferences()
        self.match()


class Simulation(object):
    def __init__(self, marketplace_params: list, n_iters: int):
        self.marketplace_params = marketplace_params
        self.n_iters = n_iters
        self.m_specs = []

        self.summary_stats = defaultdict(list)

    def run(self):
        for m_spec in tqdm(self.marketplace_params):
            raw_stats = defaultdict(list)
            for i in range(self.n_iters):
                m = Marketplace(**m_spec)
                m.run_marketplace()

                for k, v in m.get_stats().items():
                    raw_stats[k].append(v)

            self.m_specs.append(m.specification)
            for k, v in raw_stats.items():
                self.summary_stats[k].append(np.mean(v))

            # Add in nans for potentially not existing things
            for o in [
                "sim_a_rulebreaker_mean",
                "sim_a_rulebreaker_std",
                "sim_e_rulebreaker_mean",
                "sim_e_rulebreaker_std",
                "sim_a_rulefollower_mean",
                "sim_a_rulefollower_std",
                "sim_e_rulefollower_mean",
                "sim_e_rulefollower_std",
            ]:
                if o not in raw_stats:
                    self.summary_stats[o].append(np.nan)

    def to_df(self):
        # Takes a name for whatever is varying + the values and returns a dataframe with results
        data = defaultdict(list)
        for d in self.m_specs:
            for key, value in d.items():
                if value is None:
                    value = "None"
                data[key].append(value)
        for k, v in self.summary_stats.items():
            data[k] = v

        return pd.DataFrame(data)

    def to_melted_df(self):
        df = self.to_df()
        value_vars = [
            col
            for col in df.columns
            if col.startswith("sim_") or col.startswith("alg_")
        ]
        id_vars = [col for col in df.columns if col not in value_vars]

        df = pd.melt(
            df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="outcome",
            value_name="value",
        )

        df["method"] = df.outcome.apply(
            lambda x: "simulated" if "sim" in x else "algorithmic"
        )
        df["type"] = df.outcome.apply(
            lambda x: "employer" if "_e_" in x else "applicant"
        )
        df["method_type"] = df.method + "-" + df.type
        df["stat"] = df.outcome.apply(lambda x: "mean" if "mean" in x else "std")
        # df = df.drop(columns=["temp"])
        id_vars += ["method", "type", "method_type", "outcome"]
        df = df.pivot_table(index=id_vars, columns="stat", values="value").reset_index()
        return df
