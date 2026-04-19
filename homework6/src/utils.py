import torch
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA


def _to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def negative_sampling(data, num_samples, excluded_edge_index=None, device="cpu"):
    node_types = list(getattr(data, "node_types", []))
    if node_types and set(node_types) == {"drug", "disease"}:
        src_candidates = [i for i, node_type in enumerate(node_types) if node_type == "drug"]
        dst_candidates = [i for i, node_type in enumerate(node_types) if node_type == "disease"]
        mirror_edges = False
    else:
        src_candidates = list(range(data.num_nodes))
        dst_candidates = list(range(data.num_nodes))
        mirror_edges = True

    if not src_candidates or not dst_candidates:
        raise ValueError("Unable to sample negatives because the candidate node sets are empty.")

    if excluded_edge_index is None:
        excluded_edge_index = data.edge_index

    existing_edges = {
        (int(src), int(dst))
        for src, dst in zip(
            excluded_edge_index[0].detach().cpu().tolist(),
            excluded_edge_index[1].detach().cpu().tolist(),
        )
    }
    if mirror_edges:
        existing_edges |= {(dst, src) for src, dst in existing_edges}

    neg_src, neg_dst = [], []
    sampled_edges = set(existing_edges)
    max_attempts = max(num_samples * 20, 1000)
    attempts = 0

    while len(neg_src) < num_samples and attempts < max_attempts:
        src = src_candidates[torch.randint(len(src_candidates), (1,), device=device).item()]
        dst = dst_candidates[torch.randint(len(dst_candidates), (1,), device=device).item()]
        attempts += 1

        if src == dst:
            continue
        if (src, dst) in sampled_edges:
            continue

        neg_src.append(src)
        neg_dst.append(dst)
        sampled_edges.add((src, dst))
        if mirror_edges:
            sampled_edges.add((dst, src))

    if len(neg_src) != num_samples:
        raise ValueError("Unable to sample the requested number of negative edges without collisions.")

    return torch.tensor([neg_src, neg_dst], dtype=torch.long, device=device)


def stratified_split_edges(data, test_size=0.2, val_size=0.1, seed=42):
    if not 0.0 < test_size < 1.0 or not 0.0 < val_size < 1.0 or test_size + val_size >= 1.0:
        raise ValueError("test_size and val_size must be between 0 and 1, and their sum must be less than 1.")

    edge_index_np = data.edge_index.cpu().numpy()
    edge_type_np = data.edge_type.cpu().numpy()

    all_indices = np.arange(edge_index_np.shape[1])
    train_idx, temp_idx, _, temp_edge_types = train_test_split(
        all_indices,
        edge_type_np,
        test_size=test_size + val_size,
        stratify=edge_type_np,
        random_state=seed,
    )

    relative_test_size = test_size / (test_size + val_size)
    val_idx, test_idx, _, _ = train_test_split(
        temp_idx,
        temp_edge_types,
        test_size=relative_test_size,
        stratify=temp_edge_types,
        random_state=seed,
    )

    def to_tensor(indices):
        edge_index = torch.tensor(edge_index_np[:, indices], dtype=torch.long)
        edge_type = torch.tensor(edge_type_np[indices], dtype=torch.long)
        return edge_index, edge_type

    return *to_tensor(train_idx), *to_tensor(val_idx), *to_tensor(test_idx)


def print_graph_stats(edge_index, edge_type=None, num_nodes=None):
    print(f"Number of edges: {edge_index.size(1)}")
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    print(f"Number of nodes: {num_nodes}")

    degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)
    print(f"Average node degree: {degrees.float().mean():.2f}")
    print(f"Max node degree: {degrees.max().item()}")

    if edge_type is not None:
        edge_type_counts = Counter(torch.as_tensor(edge_type).detach().cpu().tolist())
        print("Edge types distribution:")
        for etype, count in edge_type_counts.items():
            print(f"  Type {etype}: {count} edges")

def plot_node_degree_distribution(edge_index, num_nodes=None, title="Node Degree Distribution"):
    if num_nodes is None:
        num_nodes = int(edge_index.max()) + 1
    degrees = torch.bincount(edge_index.view(-1), minlength=num_nodes)

    plt.figure()
    plt.hist(_to_numpy(degrees), bins=30, log=True)
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes (log scale)")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_edge_type_distribution(edge_type, title="Edge Type Distribution"):
    edge_type_counts = Counter(torch.as_tensor(edge_type).detach().cpu().tolist())
    labels, values = zip(*sorted(edge_type_counts.items()))

    plt.figure()
    plt.bar(labels, values)
    plt.xlabel("Edge Type ID")
    plt.ylabel("Count")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_embeddings(embeddings, method="pca"):
    embeddings = _to_numpy(embeddings)
    if embeddings.shape[0] < 2:
        raise ValueError("Need at least two nodes to visualize embeddings.")

    if method == "pca":
        reduced = PCA(n_components=2).fit_transform(embeddings)
    elif method == "tsne":
        perplexity = min(30, max(1, embeddings.shape[0] - 1))
        reduced = TSNE(
            n_components=2,
            perplexity=perplexity,
            learning_rate=200,
            init="pca",
            random_state=42,
        ).fit_transform(embeddings)
    else:
        raise ValueError("method must be 'pca' or 'tsne'.")

    fig, ax = plt.subplots()
    ax.scatter(reduced[:, 0], reduced[:, 1], s=8)
    ax.set_title(f"{method.upper()} of Embeddings")
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.tight_layout()
    return fig
