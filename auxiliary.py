r"""Auxiliary functions for the lecture on dealing with selection bias."""
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def simulate_sample(num_persons, rho):
    r"""Simulate sample.

    This function simulates a sample from the simplified Roy model used during the lecture.

    Parameters
    ----------
    num_persons : int
        Number of persons to simulate.

    rho : float
        Impact of ability on potential outcomes.

    Returns
    -------

    df: pandas.DataFrame
        Data frame with observed and unobserved information about each person.

    """
    np.random.seed(123)

    index = range(num_persons)
    tuples = list(product(["observable"], ["y", "d", "t"]))
    tuples += list(product(["unobservable"], ["y1", "y0", "u1", "u0"]))
    columns = pd.MultiIndex.from_tuples(tuples)

    df = pd.DataFrame(columns=columns, index=index)
    df.index.name = "person"

    for i in range(num_persons):

        # Sample characteristics
        u1, u0 = np.random.normal(scale=0.1, size=2)
        t = np.random.uniform(-0.5, 0.5)

        # Construct potential outcomes and treatment status
        y1 = rho * t + u1
        y0 = u0
        d = t > 0.0

        # Derive observed outcome
        y = d * y1 + (1 - d) * y0

        # Collect all information in dataset
        df.loc[i] = y, d, t, y1, y0, u1, u0

    # Proper types for input columns
    df = df.astype(np.float)
    df = df.astype({("observable", "d"): np.int})

    return df


def plot_joint_distribution(num_persons, rho):
    r"""Plot joint distribution.

    This function plots the joint distribution of ability and the observed outcome for a
    simulated sample of `num_agents` and given value of `rho`.

    Parameters
    ----------
    num_persons : int
        Number of persons to simulate.

    rho : float
        Impact of ability on potential outcomes.

    Returns
    -------
    None

    """
    df_obs = simulate_sample(num_persons, rho)["observable"]
    h = sns.jointplot(df_obs["t"], df_obs["y"], xlim=(-0.5, 0.5), ylim=(-0.5, 0.5))
    h.set_axis_labels(r"$\theta$", "$Y$", fontsize=16, fontweight="bold")
    h.fig.text(0.6, 0.3, r"$\bar{\beta}$ = 0.00", size=20)

    estimate = df_obs.query("d == 1")["y"].mean() - df_obs.query("d == 0")["y"].mean()
    h.fig.text(0.6, 0.225, r"$\hat{\beta}$ =" + f"{estimate:5.2f}", size=20)
    h.fig.text(0.6, 0.15, r"$\rho$ = 0.00", size=20)


def plot_distribution_benefit(num_persons, rho):
    r"""Plot marginal distribution.

    This function plots the the marginal distribution of benefits for a simulated sample of
    `num_persons`and given value of `rho`.

    Parameters
    ----------
    num_persons : int
        Number of persons to simulate.

    rho : float
        Impact of ability on potential outcomes.

    Returns
    -------
    None

    """
    df_unobs = simulate_sample(num_persons, rho)["unobservable"]

    fig, ax = plt.subplots()
    sns.distplot(df_unobs["y1"] - df_unobs["y0"], ax=ax)
    ax.set_xlabel(r"$\Delta$")
    ax.set_ylabel(r"Density")
    ax.set_yticklabels([])
    ax.text(0.4, 2.5, r"$\bar{\beta} = 0.0$", size=20)
    ax.text(0.4, 2.0, fr"$\rho = {rho}$", size=20)
    ax.set_xlim([-0.8, 0.8])
    ax.set_ylim([0.0, 3.5])
    ax.axvline(x=0, color="grey", linestyle="--")
