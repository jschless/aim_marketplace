import pandas as pd
from collections import defaultdict

from marketplace import *


class Simulation(object):
    def __init__(self, marketplace_params: list, n_iters: int):
        """
        full_stats: {stat_name -> list where each item is results from a particular marketplace instantiation ->
                     list of vector for each iteration where each vector is the preference received for all candidates}
        vec_stats: {stat_name -> list where each item is results from a particular marketplace instantiation ->
                    list of summary stats for each iteration}
        summary_stats: {stat_name -> list of length # specifications -> average of statistic across all iterations}
        """
        self.marketplace_params = marketplace_params
        m = Marketplace(**marketplace_params[0])
        self.marketplace_description = m.specification
        self.n_iters = n_iters
        self.trials = []
        # stores data from all trials. Each trial is unique based on its specification and it's iter

    def run(self):
        trial = {}
        for m_spec in tqdm(self.marketplace_params):
            for i in range(self.n_iters):
                m = Marketplace(**m_spec)
                m.run_marketplace()
                trial["specification"] = m.specification

                _, vectors = m.get_stats()

                for k, v in vectors.items():
                    mean = np.mean(v) if v.size > 0 else np.nan
                    std = np.std(v) if v.size > 0 else np.nan
                    trial[k + "_mean"] = mean
                    trial[k + "_std"] = std
                    trial[k + "_vec"] = v

                trial["iter"] = i
                self.trials.append(trial.copy())

    def to_long_df(self):
        prefixes = [
            "sim_e_",
            "sim_e_rulebreaker_",
            "sim_e_truther_",
            "sim_e_rulefollower_",
            "sim_e_liar_",
            "sim_a_",
            "sim_a_rulebreaker_",
            "sim_a_truther_",
            "sim_a_rulefollower_",
            "sim_a_liar_",
            "alg_a_",
            "alg_e_",
        ]
        data = []
        for t in self.trials:
            row = {}
            for param, value in t["specification"].items():
                if value is None:
                    value = "None"
                row[param] = value

            for pref in prefixes:
                if pref + "mean" in t:
                    row["outcome"] = pref[:-1]
                    row["trial_mean"] = t[pref + "mean"]
                    row["trial_std"] = t[pref + "std"]
                    data.append(row.copy())

        df = pd.DataFrame(data)

        # add some nice columns
        df["Party"] = df.outcome.apply(lambda x: "Unit" if "_e" in x else "Officer")

        def labeller(di, word):
            for k, v in di.items():
                if k in word:
                    return v
            return None

        rule_follower_dict = {
            "follower": "Rule Follower",
            "breaker": "Rule Breaker",
            "alg": "Gale-Shapley",
        }
        df["Rule Following Status"] = df.outcome.apply(
            lambda x: labeller(rule_follower_dict, x)
        )

        liar_dict = {"liar": "Liar", "truther": "Truth Teller", "alg": "Gale-Shapley"}
        df["Lying Status"] = df.outcome.apply(lambda x: labeller(liar_dict, x))

        alg_dict = {"alg": "Algorithmic", "sim": "Simulated"}
        df["alg_or_sim"] = df.outcome.apply(lambda x: labeller(alg_dict, x))

        renamer = {
            "trial_mean": "Average Preference Received",
            "type": "Side of Market",
            "rule_following": "Adherence to Rules",
            "liar": "Honesty of Commitments",
        }

        return df
