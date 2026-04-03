import numpy as np

from collections import defaultdict

from adeft.disambiguate import AdeftDisambiguator
from adeft.modeling.classify import load_model_info

from indra_db_lite.api import get_text_ref_ids_for_agent_text
from indra_db_lite.api import get_plaintexts_for_text_ref_ids

from opaque.nlp.models import GroundingAnomalyDetector
from opaque.stats import highest_density_interval

from adeft_indra.results import ResultsManager



if __name__ == "__main__":
    good = []
    results_db = ResultsManager("/home/birbir/extra_storage/auto_adeft_models_3.db")
    for model_name, model_info in results_db.items():
        if "Failed" in model_info:
            continue

        grounding_dict = model_info["grounding_dict"]
        names = model_info["names"]
        names = {key: str(val) for key, val in names.items()}
        adeft_model = load_model_info(model_info["adeft_model"])
        if not adeft_model.pos_labels:
            continue

        print(f"processing model {model_name}")
        disamb = AdeftDisambiguator(adeft_model, grounding_dict, names)
        all_labels = disamb.labels
        classifier_labels = disamb.classifier.stats["label_distribution"]
        pos_labels = disamb.classifier.pos_labels

        # need to do this again here because we left some steps out in
        # the main run
        trids = set()
        for shortform in disamb.shortforms:
            trids.update(get_text_ref_ids_for_agent_text(shortform))
        trids = list(trids)
        texts = list(get_plaintexts_for_text_ref_ids(trids, contains=disamb.shortforms))
        disambiguations = disamb.disambiguate(texts)
        dp_idx_dict = defaultdict(list)
        no_dp_idx_dict = defaultdict(list)
        for i, res in enumerate(disambiguations):
            grounding = res["decision"]
            defining_patterns = res["defining_patterns"]
            from_defining_pattern = (
                defining_patterns is not None and len(defining_patterns) == 1
            )
            if from_defining_pattern:
                dp_idx_dict[grounding].append(i)
            else:
                no_dp_idx_dict[grounding].append(i)

        with_dp_counts = {grounding: len(val) for grounding, val in dp_idx_dict.items()}
        no_dp_counts = {grounding: len(val) for grounding, val in no_dp_idx_dict.items()}

        training_label_intervals = {}
        all_data_intervals = {}
        frequencies = {}
        for grounding, ad_model_info in model_info["ad_models"].items():
            if ad_model_info is None:
                continue
            shape_params = ad_model_info["shape_params"]
            sens_a, sens_b = shape_params["sens_alpha"], shape_params["sens_beta"]
            spec_a, spec_b = shape_params["spec_alpha"], shape_params["spec_beta"]
            model = GroundingAnomalyDetector.load_model_info(ad_model_info["model"])

            idx = dp_idx_dict[grounding]
            preds = model.predict([texts[i] for i in idx])
            n = len(preds)
            t = np.sum(preds == -1.0)
            interval = highest_density_interval(
                n, t, sens_a, sens_b, spec_a, spec_b, rng=1729
            )
            interval = (1.0 - interval[1], 1.0-interval[0])
            training_label_intervals[grounding] = interval

            idx = no_dp_idx_dict[grounding]
            preds = model.predict([texts[i] for i in idx])
            m = len(preds)
            u = np.sum(preds == -1.0)
            interval = highest_density_interval(
                n + m, t + u, sens_a, sens_b, spec_a, spec_b, rng=1729
            )
            interval = (1.0 - interval[1], 1.0-interval[0])
            all_data_intervals[grounding] = interval

            
            frequencies[grounding] = [n, t, m, u]

        intervals = model_info["intervals"]
        intervals = {
            key: val["standard"]["precision"]
            for key, val in intervals.items()
            if val is not None
            and key in classifier_labels
        }
        disamb_info = {
            "grounding_dict": disamb.grounding_dict,
            "names": disamb.names,
            "model": disamb.classifier.get_model_info(),
        }
        good.append({
            "disamb": disamb_info,
            "classifier_labels": classifier_labels,
            "pos_labels": pos_labels,
            "prediction_intervals": intervals,
            "training_label_intervals": training_label_intervals,
            "all_data_intervals": all_data_intervals,
            "with_dp_counts": with_dp_counts,
            "no_dp_counts": no_dp_counts,
            "frequencies": frequencies,
        })
