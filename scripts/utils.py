import os
import pickle
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

def save_embeddings(embeddings, filepath):
    """
    Save node embeddings to a pickle file.

    Parameters:
        embeddings (dict): Node ID -> embedding vector
        filepath (str): Path to save the file
    """
    with open(filepath, 'wb') as f:
        pickle.dump(embeddings, f)
    print(f"[✔] Embeddings saved to {filepath}")

def load_embeddings(filepath):
    """
    Load node embeddings from a pickle file.

    Parameters:
        filepath (str): Path to the pickle file

    Returns:
        dict: Node ID -> embedding vector
    """
    with open(filepath, 'rb') as f:
        embeddings = pickle.load(f)
    print(f"[✔] Embeddings loaded from {filepath}")
    return embeddings

def save_graph(G, filepath):
    """
    Save a NetworkX graph to a .gpickle file.

    Parameters:
        G (networkx.Graph): Graph object
        filepath (str): Path to save the graph
    """
    nx.write_gpickle(G, filepath)
    print(f"[✔] Graph saved to {filepath}")

def load_graph(filepath):
    """
    Load a NetworkX graph from a .gpickle file.

    Parameters:
        filepath (str): Path to the graph file

    Returns:
        networkx.Graph: Loaded graph
    """
    G = nx.read_gpickle(filepath)
    print(f"[✔] Graph loaded from {filepath}")
    return G

def plot_tsne(embeddings, title="Node Embeddings (t-SNE)", figsize=(10, 7)):
    """
    Visualize high-dimensional embeddings using t-SNE.

    Parameters:
        embeddings (dict): Node ID -> embedding vector
        title (str): Plot title
        figsize (tuple): Figure size
    """
    from sklearn.manifold import TSNE
    import numpy as np

    X = list(embeddings.values())
    labels = list(embeddings.keys())

    tsne = TSNE(n_components=2, random_state=42)
    X_2d = tsne.fit_transform(X)

    plt.figure(figsize=figsize)
    plt.scatter(X_2d[:, 0], X_2d[:, 1], s=10, alpha=0.7)
    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)
    plt.show()
