import pandas as pd
from biopandas.pdb import PandasPdb
from prody import parsePDBHeader




def read_pdb_to_dataframe(
    pdb_path,
    model_index: int = 1,
    parse_header: bool = True,
    ) -> pd.DataFrame:
    """
    Read a PDB file, and return a Pandas DataFrame containing the atomic coordinates and metadata.

    Args:
        pdb_path (str, optional): Path to a local PDB file to read. Defaults to None.
        model_index (int, optional): Index of the model to extract from the PDB file, in case
            it contains multiple models. Defaults to 1.
        parse_header (bool, optional): Whether to parse the PDB header and extract metadata.
            Defaults to True.

    Returns:
        pd.DataFrame: A DataFrame containing the atomic coordinates and metadata, with one row
            per atom
    """
    atomic_df = PandasPdb().read_pdb(pdb_path)
    if parse_header:
        header = parsePDBHeader(pdb_path)
    else:
        header = None
    atomic_df = atomic_df.get_model(model_index)
    if len(atomic_df.df["ATOM"]) == 0:
        raise ValueError(f"No model found for index: {model_index}")

    return pd.concat([atomic_df.df["ATOM"], atomic_df.df["HETATM"]]), header

from graphein.protein.graphs import label_node_id

def process_dataframe(df: pd.DataFrame, granularity='CA') -> pd.DataFrame:
    """
    Process a DataFrame of protein structure data to reduce ambiguity and simplify analysis.

    This function performs the following steps:
    1. Handles alternate locations for an atom, defaulting to keep the first one if multiple exist.
    2. Assigns a unique node_id to each residue in the DataFrame, using a helper function label_node_id.
    3. Filters the DataFrame based on specified granularity (defaults to 'CA' for alpha carbon).

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing protein structure data to process. It is expected to contain columns 'alt_loc' and 'atom_name'.
        
    granularity : str, optional
        The level of detail or perspective at which the DataFrame should be analyzed. Defaults to 'CA' (alpha carbon).
    """
    # handle the case of alternative locations,
    # if so default to the 1st one = A
    if 'alt_loc' in df.columns:
      df['alt_loc'] = df['alt_loc'].replace('', 'A')
      df = df.loc[(df['alt_loc']=='A')]
    df = label_node_id(df, granularity)
    df = df.loc[(df['atom_name']==granularity)]
    return df


from graphein.protein.graphs import initialise_graph_with_metadata
from graphein.protein.graphs import add_nodes_to_graph
from graphein.protein.visualisation import plotly_protein_structure_graph
from PIL import Image
import networkx as nx

def take_care(pdb_path):
    

    df, header = read_pdb_to_dataframe(pdb_path)
    process_df = process_dataframe(df)

    g = initialise_graph_with_metadata(protein_df=process_df, # from above cell
                                        raw_pdb_df=df, # Store this for traceability
                                        pdb_code = '3nir', #and again
                                        granularity = 'CA' # Store this so we know what kind of graph we have
                                        )
    g = add_nodes_to_graph(g)

    
    def add_backbone_edges(G: nx.Graph) -> nx.Graph:
        # Iterate over every chain
        for chain_id in G.graph["chain_ids"]:
            # Find chain residues
            chain_residues = [
                (n, v) for n, v in G.nodes(data=True) if v["chain_id"] == chain_id
            ]
            # Iterate over every residue in chain
            for i, residue in enumerate(chain_residues):
                try:
                    # Checks not at chain terminus
                    if i == len(chain_residues) - 1:
                        continue
                    # Asserts residues are on the same chain
                    cond_1 = ( residue[1]["chain_id"] == chain_residues[i + 1][1]["chain_id"])
                    # Asserts residue numbers are adjacent
                    cond_2 = (abs(residue[1]["residue_number"] - chain_residues[i + 1][1]["residue_number"])== 1)

                    # If this checks out, we add a peptide bond
                    if (cond_1) and (cond_2):
                        # Adds "peptide bond" between current residue and the next
                        if G.has_edge(i, i + 1):
                            G.edges[i, i + 1]["kind"].add('backbone_bond')
                        else:
                            G.add_edge(residue[0],chain_residues[i + 1][0],kind={'backbone_bond'},)
                except IndexError as e:
                    print(e)
        return G

    g = add_backbone_edges(g)

    

    p = plotly_protein_structure_graph(
        g,
        colour_edges_by="kind",
        colour_nodes_by="seq_position",
        label_node_ids=False,
        plot_title="Backbone Protein Graph",
        node_size_multiplier=1,
    )
    image_file = "protein_graph.png"
    p.write_image(image_file, format='png')


    # Load the PNG image into a PIL image
    image = Image.open(image_file)

    return image