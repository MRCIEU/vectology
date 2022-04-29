import pandas as pd
from nxontology import NXOntology


def create_efo_nxo(df: pd.DataFrame, child_col: str, parent_col: str) -> NXOntology:
    nxo: NXOntology = NXOntology()

    edges = []
    for i, row in df.iterrows():
        child = row[child_col]
        parent = row[parent_col]
        edges.append((parent, child))
    nxo.graph.add_edges_from(edges)
    return nxo


def create_nx(efo_rel_df: pd.DataFrame) -> NXOntology:
    efo_nx = create_efo_nxo(
        df=efo_rel_df, child_col="efo.id", parent_col="parent_efo.id"
    )
    efo_nx.freeze()
    return efo_nx
