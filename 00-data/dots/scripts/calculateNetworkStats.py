# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: ploomber
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
#   ploomber:
#     injected_manually: true
# ---

# %% tags=["parameters"]
upstream = ['dataCollect'] # this means: execute raw.py, then clean.py
product = None


# %% tags=["injected-parameters"]
# Parameters
upstream = {"dataCollect": {"nb": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\dataCollect.ipynb", "files": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\raw", "DOTS": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\clean\\DOTS.csv"}}
product = {"nb": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\calculateNetworkStats.ipynb", "DOTSnetStats": "C:\\Users\\yosty\\Desktop\\Desktop_Folder\\14 - git\\timeSeriesDOTS\\ploomber\\dots\\00-data\\clean\\DOTSnetStats.csv"}


# %%

import pandas as pd
import os
import numpy as np
import networkx as nx


from networkx.algorithms.approximation.connectivity import node_connectivity
from networkx.algorithms.bridges import has_bridges
from networkx.algorithms.link_analysis.pagerank_alg import pagerank, pagerank_numpy
from networkx.algorithms.approximation.connectivity import node_connectivity
from networkx.algorithms.bridges import has_bridges


# %%

data=pd.read_csv(upstream['dataCollect']['DOTS'])

keepCols=['series_code', 'period', 'value', 'REF_AREA', 'INDICATOR', 'COUNTERPART_AREA', 'Reference Area', 'Counterpart Reference Area']

cleanData = data[keepCols]

# %%
# cleanData['INDICATOR'].unique()
# cleanData['Reference Area'].unique()
# cleanData['Counterpart Reference Area'].unique()

cleanData.rename(columns={'Counterpart Reference Area':'CounterpartReferenceArea',
        'Reference Area':'ReferenceArea'}, inplace=True)

# %%

# drop groups
# clean nan filled economies

wideData = (cleanData
    .pivot_table(
        index=['period', 'CounterpartReferenceArea'],
        columns='ReferenceArea',
        values='value'))


# %%

# remove groups
# decided to keep 'Special Categories and Economic Zones' - sounds interesting
# and wouldn't cause overlap to include i think
removeList = [
'Export earnings: fuel',
'Export earnings: nonfuel',
'Middle East, North Africa, Afghanistan, and Pakistan',
'Emerging and Developing Europe',
'Western Hemisphere',
'Western Hemisphere not allocated',
'EU (Member States and Institutions of the European Union) changing composition',
'Euro Area (Member States and Institutions of the Euro Area) changing composition',
'Europe',
'Europe not allocated',
'Africa',
'Africa not allocated',
'Sub-Saharan Africa',
'Middle East',
'Middle East and Central Asia not specified',
'Other Countries n.i.e. (IMF)',
'Advanced Economies (IMF)',
'Emerging and Developing Countries',
'Developing Asia (IMF)',
'Middle East and Central Asia',
'Belgo-Luxembourg Economic Union',
'Community of Independent States (CIS)',
'Asia not allocated',
'Former U.S.S.R.',
'All Countries, excluding the IO']


dataLong = (wideData
    .melt(ignore_index=False)
    .query('CounterpartReferenceArea not in @removeList')
    .query('ReferenceArea not in @removeList')
)

print("Lost: ", np.round((wideData.melt().shape[0] - dataLong.shape[0]) / wideData.melt().shape[0], 2) * 100, "% rows due to dropping groups")

wideData2 = dataLong.pivot_table(index=['period', 'CounterpartReferenceArea'],
                                columns='ReferenceArea',
                                values='value')

dataLong.reset_index(inplace=True)
dataLong.rename(columns={'value':'weight'}, inplace=True)


# #%%
# # nans

# colna = wideData2.isna().sum()
# colna = colna / wideData2.shape[0]
# colna.nlargest(20)

# #%%
# colna.nsmallest(20)



# %%


dates = dataLong['period'].unique()

stats=[]
for date in dates:

    tempData=dataLong.query('period == @date')
    tempData.dropna(axis=0, inplace=True)

    G = nx.from_pandas_edgelist(df = tempData,
                                        source = "ReferenceArea",
                                        target = "CounterpartReferenceArea",
                                        edge_attr = "weight",
                                        create_using = nx.DiGraph())

    tempdf = pd.DataFrame( dict(
                    # key data
                    # typf of graph
                    #graphType = G.
                    PERIOD = date,


                    # --------------------------------MICRO

                    ## ---- Centrality
                    # returns a dict of node's degree
                    DEGREE = dict(G.degree),
                    IN_DEGREE = dict(G.in_degree),
                    OUT_DEGREE = dict(G.out_degree),

                    # fraction of nodes a node is connected to
                    DEGREE_CENTRALITY = nx.degree_centrality(G),
                    IN_DEGREE_CENTRALITY = nx.in_degree_centrality(G),
                    OUT_DEGREE_CENTRALITY = nx.out_degree_centrality(G),
                    AVG_NEIGHBOR_DEGREE = nx.average_neighbor_degree(G),

                    # centrality based on importance of edges
                    PAGERANK = pagerank(G, weight = 'weight'),
                    PAGERANK_NUMPY = pagerank_numpy(G, weight = 'weight'),


                    # centrality based on neighbors
                    #EIGENVECTOR_CENTRAL = nx.eigenvector_centrality_numpy(G),
                    # generalization of eigen centrality
                    KATZ = nx.katz_centrality_numpy(G),
                    CLOSENESS_CENTRALITY = nx.closeness_centrality(G),
                    BETWEENNESS_CENTRALITY = nx.betweenness_centrality(G),

                    ## ---- Paths

                    ## ---- Clustering
                    # node clustering scores
                    CLUSTCOEF = nx.clustering(G),



                    #-----------------------------------MACRO
                    ##  --- Size
                    NUM_NODES = G.number_of_nodes(),
                    NUM_EDGES = G.number_of_edges(),
                    TOTAL_NET_VALUE = tempData['weight'].sum(),

                    ## ----- Connectivity
                    CONNECTIVITY = node_connectivity(G),

                    # edge whose removal causes the number of connected components of the graph to increase
                    HAS_BRIDGE = has_bridges(nx.Graph(G)),

                    ## ---- Clustering
                    # ego net clusterting score
                    # graph clustering score
                    AVERAGECLUSTCOEF = nx.average_clustering(G),
                    TRIANGLES = nx.triangles(nx.Graph(G)),

                ))

    stats.append(tempdf)
    # print(date)


statDF = pd.concat(stats)
statDF.reset_index(inplace=True)
statDF.to_csv(str(product['DOTSnetStats']))

