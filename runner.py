import os
import pickle
import itertools

import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("darkgrid")

from simulation import *


def generate_plot_description(sim, ind_var):
    description = ""
    m = sim.marketplace_description
    if ind_var != "market_size":
        description += f"Market size: {m['market_size']}"
    if ind_var != "employer_min_acceptance_thresh":
        description += (
            f"\nMinimum Ranking to Offer/Accept: {m['employer_min_acceptance_thresh']}"
        )
    if ind_var != "employer_rule_follower_rate" and ind_var != "rule_following":
        description += (
            f"\nEmployer Rule Follower Rate: {m['employer_rule_follower_rate']}"
        )
    if ind_var != "applicant_rule_follower_rate" and ind_var != "rule_following":
        description += (
            f"\nApplicant Rule Follower Rate: {m['applicant_rule_follower_rate']}"
        )
    if ind_var != "employer_lie_rate" and ind_var != "liar":
        description += f"\nEmployer Liar Rate: {m['employer_lie_rate']}"
    if ind_var != "applicant_lie_rate" and ind_var != "liar":
        description += f"\nApplicant Liar Rate: {m['applicant_lie_rate']}"

    description += f"\n# Iterations: {sim.n_iters}"

    return description


def plot(sim, x_var, style, xlabel, output_file, output_dir="./plots", prefix=""):
    fig, ax = plt.subplots()

    sns.lineplot(
        data=sim.to_long_df(), x=x_var, y="trial_mean", hue="type", style=style
    )
    plt.xlabel(xlabel)
    plt.ylabel("Mean Received Preference")
    desc = generate_plot_description(sim, x_var)
    plt.gcf().text(
        0.5,
        0.2,  # -0.1,
        desc,
        ha="center",
        va="top",
        fontsize=8,
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.subplots_adjust(bottom=0.3)

    with open(os.path.join(output_dir, f"{prefix}{output_file}.pkl"), "wb") as f:
        pickle.dump(fig, f)
    plt.savefig(os.path.join(output_dir, f"{prefix}{output_file}"))

    plt.clf()


def run_all():
    # Should run a bunch of simulations, plot them, and save the figures
    #
    market_sizes = list(range(25, 201, 25))
    acceptance_thresholds = [None, 10, 20]
    rule_follower_rates = [0.1 * i for i in range(11)]
    lie_rates = [0.1 * i for i in range(11)]
    n_iters = 100
    results_dir = "/Users/joeschlessinger/Programming/marketplace/plots"

    for a_t, market_size in itertools.product(acceptance_thresholds, market_sizes):
        default_marketplace_args = {
            "employer_min_acceptance_thresh": a_t,
            "applicant_min_acceptance_thresh": a_t,
            "market_size": market_size,
        }
        plot_prefix = ""
        plot_dir = os.path.join(
            results_dir, f"acceptance_thresh={a_t}_market_size={market_size}_"
        )
        if not os.path.exists(plot_dir):
            os.mkdir(plot_dir)

        # Vary market size
        s = Simulation(
            [{"market_size": i, **default_marketplace_args} for i in market_sizes],
            n_iters,
        )
        s.run()
        plot(
            s,
            "market_size",
            "alg_or_sim",
            "Market Size",
            "market_size.png",
            prefix=plot_prefix,
            output_dir=plot_dir,
        )

        # Vary both rule follower rates
        s = Simulation(
            [
                {
                    "market_size": 50,
                    "employer_rule_follower_rate": i,
                    "applicant_rule_follower_rate": i,
                    **default_marketplace_args,
                }
                for i in rule_follower_rates
            ],
            n_iters,
        )
        s.run()
        plot(
            s,
            "employer_rule_follower_rate",
            "rule_following",
            "Rule Follower Rate",
            "equal_rule_following.png",
            prefix=plot_prefix,
            output_dir=plot_dir,
        )

        # Vary employer rule follower rates, hold applicant constant
        for i, a_r in enumerate(rule_follower_rates):
            s = Simulation(
                [
                    {
                        "market_size": 50,
                        "employer_rule_follower_rate": i,
                        "applicant_rule_follower_rate": a_r,
                        **default_marketplace_args,
                    }
                    for i in rule_follower_rates
                ],
                n_iters,
            )
            s.run()
            plot(
                s,
                "employer_rule_follower_rate",
                "rule_following",
                "Employer Rule Follower Rate",
                f"applicant_rule_following_{a_r}.png",
                prefix=plot_prefix,
                output_dir=plot_dir,
            )

            # Vary applicant rule follower rates, hold employer constant
            for i, a_r in enumerate(rule_follower_rates):
                s = Simulation(
                    [
                        {
                            "market_size": 50,
                            "employer_rule_follower_rate": a_r,
                            "applicant_rule_follower_rate": i,
                            **default_marketplace_args,
                        }
                        for i in rule_follower_rates
                    ],
                    n_iters,
                )
                s.run()
                plot(
                    s,
                    "applicant_rule_follower_rate",
                    "rule_following",
                    "Applicant Rule Follower Rate",
                    f"employer_rule_following_{a_r}.png",
                    prefix=plot_prefix,
                    output_dir=plot_dir,
                )

        # Vary both lie rates
        s = Simulation(
            [
                {
                    "market_size": 50,
                    "employer_lie_rate": i,
                    "applicant_lie_rate": i,
                    **default_marketplace_args,
                }
                for i in lie_rates
            ],
            n_iters,
        )
        s.run()
        plot(
            s,
            "employer_lie_rate",
            "liar",
            "Lie Rate",
            "equal_lying.png",
            prefix=plot_prefix,
            output_dir=plot_dir,
        )

        # Vary employer lie rates, hold applicant constant
        for i, a_r in enumerate(lie_rates):
            s = Simulation(
                [
                    {
                        "market_size": 50,
                        "employer_lie_rate": i,
                        "applicant_lie_rate": a_r,
                        **default_marketplace_args,
                    }
                    for i in lie_rates
                ],
                n_iters,
            )
            s.run()
            plot(
                s,
                "employer_lie_rate",
                "liar",
                "Employer Lie Rate",
                f"applicant_lying_{a_r}.png",
                prefix=plot_prefix,
                output_dir=plot_dir,
            )

        # Vary applicant lie rates, hold employer constant
        for i, a_r in enumerate(lie_rates):
            s = Simulation(
                [
                    {
                        "market_size": 50,
                        "employer_lie_rate": a_r,
                        "applicant_lie_rate": i,
                        **default_marketplace_args,
                    }
                    for i in lie_rates
                ],
                n_iters,
            )
            s.run()
            plot(
                s,
                "applicant_lie_rate",
                "liar",
                "Applicant Lie Rate",
                f"employer_lying_{a_r}.png",
                prefix=plot_prefix,
                output_dir=plot_dir,
            )


# s = Simulation([{"market_size": i} for i in [10, 20, 30]], 20)
# s.run()
# print(generate_plot_description(s, "E"))
# plot(s, "market_size", "alg_or_sim", "Market Size", "market_size.png")
run_all()
