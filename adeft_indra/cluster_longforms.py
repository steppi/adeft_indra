import torch
import numpy as np
import networkx as nx

from collections import defaultdict
from itertools import combinations

from sentence_transformers import SentenceTransformer



from adeft_indra.training import (
    adeftify, auto_ground_longforms, bio_ont, mesh_term_of_interest
)



def get_names_and_pos_labels(grounding_dict):
    names = {}
    for grounding_map in grounding_dict.values():
        for grounding in grounding_map.values():
            if grounding != "ungrounded":
                names[grounding] = str(bio_ont.get_name(*grounding.split(":", maxsplit=1)))
    pos_labels = sorted(
        [
            label for label in names
            if label != "ungrounded"
            and not label.startswith("AMBIGUOUS")
            and not (
                label.startswith("MESH:")
                and not mesh_term_of_interest(label.split(":")[1])
            )
        ]
    )
    return names, pos_labels


class CosineSimilarity:
    def __init__(
            self,
            *,
            model=None,
            batch_size=128,
    ):
        if model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(
                "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb",
                device=device,
            )
        self.model = model
        self.batch_size = batch_size

    def __call__(self, longforms):
        with torch.inference_mode():
            embeddings = self.model.encode(
                longforms,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        return embeddings @ embeddings.T


cosine_similarity = CosineSimilarity()


def group_groundings(groundings):
    groundings = list(
        set(
            tuple(grounding.split(":", maxsplit=1))
            for grounding in groundings
            if grounding != "ungrounded"
        )
    )
    G = nx.Graph()
    N = len(groundings)
    G.add_nodes_from(groundings)
    edges = []
    for (ns1, id1), (ns2, id2) in combinations(groundings, 2):
        if bio_ont.nearest_common_descendant(ns1, id1, ns2, id2, ["isa"]) is not None:
            edges.append(((ns1, id1), (ns2, id2)))
    G.add_edges_from(edges)

    result = {}
    for component in nx.connected_components(G):
        common_descendant = None
        for ns, id_ in component:
            if common_descendant is None:
                common_descendant = (ns, id_)
            else:
                common_descendant = bio_ont.nearest_common_descendant(
                    ns, id_, *common_descendant, ["isa"]
                )
        ns2, id2 = common_descendant
        result[f"{ns2}:{id2}"] = [f"{ns}:{id_}" for ns, id_ in component]
    result = {g: key for key, val in result.items() for g in val}

    return result


def semantic_clusters(grounding_dict, *, cutoff=0.9):
    longforms = {
        (longform, f"{longform} ({shortform})", grounding)
        for shortform, grounding_map in grounding_dict.items()
        for longform, grounding in grounding_map.items()
    }
    longforms, expanded_longforms, groundings = zip(*longforms)
    sim_matrix = cosine_similarity(expanded_longforms).cpu().numpy()
    N = len(expanded_longforms)
    G = nx.Graph()
    G.add_nodes_from(range(N))

    rows, cols = np.where(np.triu(sim_matrix, k=1) >= cutoff)
    edges = [(int(r), int(c)) for r, c in zip(rows, cols)]
    G.add_edges_from(edges)

    grounding_idx = defaultdict(list)
    for i, grounding in enumerate(groundings):
        if grounding == "ungrounded":
            continue
        grounding_idx[grounding].append(i)

    for indices in grounding_idx.values():
        G.add_edges_from(combinations(indices, 2))

    longforms = np.asarray(longforms)
    groundings = np.asarray(groundings)
    components = [list(component) for component in nx.connected_components(G)]
    return [
        (longforms[list(component)], groundings[list(component)])
        for component in nx.connected_components(G)
    ]


def generate_adeft_grounding_info(shortforms):
    longforms = adeftify(shortforms)
    longform_counts = {
        lf: count for lf_list in longforms.values() for lf, count, _ in lf_list
    }
    grounding_dict, _, _ = auto_ground_longforms(longforms)
    clusters = semantic_clusters(grounding_dict)
    temp_groundings = {}
    names = {}
    for i, (longforms, groundings) in enumerate(clusters):
        new_groundings = group_groundings(groundings)
        groundings = [new_groundings.get(grounding, grounding) for grounding in groundings]
        grounding_counts = defaultdict(int)
        for longform, grounding in zip(longforms, groundings):
            grounding_counts[grounding] += longform_counts[longform]
        total_count = sum(grounding_counts.values())
        top_grounding, top_count = max(grounding_counts.items(), key=lambda x: x[1])
        if top_count / total_count < 0.5:
            local_groundings = {
                longform: f"AMBIGUOUS-{i}-{grounding}"
                for longform, grounding in zip(longforms, groundings)
            }
        else:
            local_groundings = {
                longform: grounding
                for longform, grounding in zip(longforms, groundings)
                if grounding == top_grounding
            }
        temp_groundings.update(local_groundings)

    new_grounding_dict = {
        shortform: {longform: temp_groundings[longform]
                    for longform in grounding_map
                    if longform in temp_groundings}
        for shortform, grounding_map in grounding_dict.items()
    }

    names, pos_labels = fix_groundings(new_grounding_dict)
                               
    return new_grounding_dict, names, pos_labels
