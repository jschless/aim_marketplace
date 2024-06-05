import pandas as pd
from collections import defaultdict

from marketplace import *


class Simulation(object):
    def __init__(self, marketplace_params: list, n_iters: int):
        self.marketplace_params = marketplace_params
        self.n_iters = n_iters
        self.m_specs = []
        self.full_stats = defaultdict(list)
        self.summary_stats = defaultdict(list)

    def run(self):
        for m_spec in tqdm(self.marketplace_params):
            raw_stats = defaultdict(list)
            raw_vecs = defaultdict(list)
            for i in range(self.n_iters):
                m = Marketplace(**m_spec)
                m.run_marketplace()

                # Collect stats for each iteration and store in loop-local raw_stats
                point_estimates, vectors = m.get_stats()
                for k, v in point_estimates.items():
                    raw_stats[k].append(v)

                for k, v in vectors.items():
                    raw_vecs[k].append(v)

            # Save specification of each marketplace for results presentation (and posterity)
            self.m_specs.append(m.specification)

            # The many iterations get averaged into a single value and saved
            for k, v in raw_stats.items():
                self.summary_stats[k].append(np.mean(v))

            for k, v in raw_vecs.items():
                self.full_stats[k].append(v)

            # Add in nans for potentially not existing things
            # For example, if there are no rule breakers,
            # then we will not have a rulebreaker mean or stdev
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
        # Returns dataframe where every statistic + simulation parameter is its own column
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
        # Returns melted dataframe that is good for seaborn plots
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

        id_vars += ["method", "type", "method_type", "outcome"]
        df = df.pivot_table(index=id_vars, columns="stat", values="value").reset_index()

        df["base_outcome"] = (
            df["outcome"].str.replace("_mean", "").str.replace("_std", "")
        )
        df["stat_type"] = df["outcome"].apply(
            lambda x: "mean" if "mean" in x else "std"
        )
        id_vars.append("base_outcome")
        id_vars.remove("outcome")
        pivot_df = df.pivot_table(
            index=id_vars, columns="stat_type", values=["mean", "std"]
        ).reset_index()
        pivot_df.columns = [
            "_".join(col).strip() if col[1] else col[0]
            for col in pivot_df.columns.values
        ]
        pivot_df = pivot_df.rename(
            columns={"base_outcome": "outcome", "mean_mean": "mean", "std_std": "std"}
        )

        return pivot_df
