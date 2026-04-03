import numpy as np
import pickle

from collections import Counter


from adeft.disambiguate import AdeftDisambiguator
from adeft.modeling.classify import load_model_info


with open("/home/birbir/auto_adeft_models_2026_03_25_batch2.pkl", "rb") as f:
    models = pickle.load(f)


def_keep = []
for model_info in models:
    disamb_info = model_info["disamb"]
    adeft_model = load_model_info(disamb_info["model"])
    disamb = AdeftDisambiguator(
        adeft_model,
        disamb_info["grounding_dict"],
        disamb_info["names"],
    )
    pos_labels = adeft_model.pos_labels
    all_data_intervals = model_info["all_data_intervals"]
    with_dp_counts = model_info["with_dp_counts"]
    no_dp_counts = model_info["no_dp_counts"]
    all_counts = Counter(with_dp_counts) + Counter(no_dp_counts)
    # if no distant evaluation for a label, be very conservative
    default_interval = (0.6, 1.0)
    pos_intervals = np.asarray(
        [
            all_data_intervals.get(label, default_interval)
            for label in pos_labels
        ]
    )
    weights = np.asarray([all_counts[label] for label in pos_labels], dtype=float)
    weights /= np.sum(weights)
    internal_precision = adeft_model.stats["precision"]["mean"]
    micro_averaged_interval = np.average(pos_intervals.T, weights=weights, axis=1)
    if (
            micro_averaged_interval[0] >= 0.5 and micro_averaged_interval[1] >= 0.9
            and internal_precision >= 0.75
    ):
        def_keep.append([disamb, micro_averaged_interval, all_data_intervals, all_counts])
