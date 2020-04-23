#!/usr/bin/env python
# coding: utf-8

# In[226]:


import os

import numpy as np
import pandas as pd

import networkx as nx
from networkx.drawing.nx_pylab import draw, draw_networkx_nodes, draw_networkx_edges
from networkx.classes.function import set_node_attributes
from pyvis.network import Network

import matplotlib
import mpld3
from matplotlib import pyplot as plt
from scipy.interpolate import interp1d

import statsmodels.api as sm

poll_data = pd.read_excel("encrypted.xlsx").drop("Unnamed: 0", axis=1).fillna("")

### hálózat

#### irányított

G = nx.DiGraph()

G.add_nodes_from(poll_data["ID"].unique())

edges = np.concatenate(
    poll_data.apply(
        lambda r: [
            [r["ID"], r[c]]
            for c in poll_data.columns
            if ("Kontakt" in c) and ("szám" not in c) and (r[c] != 0)
        ],
        axis=1,
    )
)

G.add_edges_from(edges)

#### irányítatlan

edges_df = (
    pd.DataFrame(edges)
    .rename(columns={0: "from", 1: "to"})
    .merge(
        pd.DataFrame(
            np.array(
                [
                    np.sort(e)
                    for e in G.edges()
                    if (e[::-1] in G.edges()) and (e[0] != e[1])
                ]
            )
        )
        .drop_duplicates()
        .reset_index(drop=True),
        how="right",
        left_on=["from", "to"],
        right_on=[0, 1],
    )
    .drop([0, 1], axis=1)
    .reset_index(drop=True)
)

G_undir = nx.Graph()

G_undir.add_nodes_from(poll_data.reset_index()["ID"].unique())

G_undir.add_edges_from(edges_df[["from", "to"]].values)

set_node_attributes(G_undir, {n: {"size": G_undir.degree(n)} for n in G_undir.nodes()})

### plusz feature-ök

#### beszédtémák megoszlása

topics = [c for c in poll_data.columns if "téma" in c]

edges_df = (
    edges_df.merge(
        poll_data[["ID"] + topics],
        how="left",
        left_on="from",
        right_on="ID",
    )
    .rename(columns={c: c.replace("-téma", "_from") for c in topics})
    .drop("ID", axis=1)
    .merge(
        poll_data[["ID"] + topics], how="left", left_on="to", right_on="ID"
    )
    .rename(columns={c: c.replace("-téma", "_to") for c in topics})
    .drop("ID", axis=1)
    .reset_index(drop=True)
)

def topic_ranker(row):

    return topics[
        np.argmax(
            [
                np.nanmean([row[c.replace("-téma", s)] for s in ["_from", "_to"]])
                for c in topics
            ]
        )
    ]

edges_df["Domináns téma"] = edges_df.apply(topic_ranker, axis=1).values

#### kontaktok

poll_data["out_degree"] = poll_data["ID"].apply(lambda n: G.out_degree(n))
poll_data["in_degree"] = poll_data["ID"].apply(lambda n: G.in_degree(n))
poll_data["mut_degree"] = poll_data["ID"].apply(lambda n: G_undir.degree(n))

### Kontakt valószínűség becslés

adjacency = (
    pd.DataFrame(
        np.concatenate(
            [
                [
                    np.sort([n, k])
                    for k in poll_data["ID"].sort_values().unique()
                    if k != n
                ]
                for n in poll_data["ID"].sort_values().unique()
            ]
        )
    )
    .drop_duplicates()
    .reset_index(drop=True)
    .rename(columns={n: "node_{}".format(n) for n in range(2)})
    .merge(
        edges_df[["from", "to"]],
        how="left",
        left_on=["node_0", "node_1"],
        right_on=["from", "to"],
    )
    .merge(
        edges_df[["from", "to"]],
        how="left",
        left_on=["node_0", "node_1"],
        right_on=["to", "from"],
    )
    .assign(edge=lambda df: (~df["from_x"].isna() | ~df["from_y"].isna()).astype(int))
    .drop(["from_x", "to_x", "from_y", "to_y"], axis=1)
    .merge(
        poll_data[["ID", "Nem", "Egyetem", "Kar", "Szint", "Pozíció", "Kommuna"]],
        how="left",
        left_on="node_0",
        right_on="ID",
    )
    .merge(
        poll_data[["ID", "Nem", "Egyetem", "Kar", "Szint", "Pozíció", "Kommuna"]],
        how="left",
        left_on="node_1",
        right_on="ID",
    )
)

for attr in ["Nem", "Egyetem", "Kar", "Szint"]:
    adjacency[attr] = (adjacency["{}_x".format(attr)] == adjacency["{}_y".format(attr)]).astype(int)

adjacency["Pozíció"] = adjacency["Pozíció_x"] + adjacency["Pozíció_y"]

adjacency["Kommuna"] = adjacency["Kommuna_x"] * adjacency["Kommuna_y"]

adjacency = adjacency.drop(
    [c for c in adjacency.columns if ("_x" in c) or ("_y" in c)], axis=1
)

def fit_logit(exog):

    logit = sm.Logit(adjacency["edge"], adjacency[exog].assign(c=1))

    results = logit.fit()
    print(results.summary())
    
X = ["Nem", "Egyetem", "Kar", "Szint", "Pozíció", "Kommuna"]
