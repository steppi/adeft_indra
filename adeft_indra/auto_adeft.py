import numpy as np

from collections import defaultdict

from adeft.disambiguate import AdeftDisambiguator
from adeft.modeling.classify import AdeftClassifier

from opaque.train import train_anomaly_detector
from opaque.nlp.models import GroundingAnomalyDetector
from opaque.stats import sample_estimated_metrics
from opaque.stats.stats import _hdi_from_sample

from adeft.nlp import english_stopwords

from .anomaly_detection.cases import get_training_cases_for_grounding
from .training import build_corpus, validate_and_refit_model
from .cluster_longforms import generate_adeft_grounding_info


from indra_db_lite.api import get_text_ref_ids_for_agent_text
from indra_db_lite.api import get_plaintexts_for_text_ref_ids


def auto_adeft(shortforms, grounding_dict, names, pos_labels):
    corpus = build_corpus(grounding_dict)
    adeft_model = validate_and_refit_model(
        shortforms, corpus, pos_labels, random_state=1729, min_class_size=10
    )
    if adeft_model is None:
        raise RuntimeError(f"Insufficient data to train model for {shortforms}")
    disamb = AdeftDisambiguator(adeft_model, grounding_dict, names)
    ad_models = {}
    groundings = [grounding for _, grounding_map in grounding_dict.items()
                  for grounding in grounding_map.values()]
    for grounding in groundings:
        if grounding == "ungrounded":
            continue
        cases = get_training_cases_for_grounding(*grounding.split(":", maxsplit=1))
        if cases is None:
            ad_models[grounding] = None
            continue
        texts = list(get_plaintexts_for_text_ref_ids(cases["train_trids"]))
        if len(texts) < 5:
            ad_models[grounding] = None
            continue
        ad_model_info = train_anomaly_detector(
            shortforms,
            texts,
            [0.2, 0.4],
            [20, 50],
            random_state=1729,
            num_mesh_texts=cases["num_mesh"],
            num_entrez_texts=cases["num_entrez"],
            predict_shape_params=True,
            stop_words=english_stopwords + ["http"],
        )
        ad_models[grounding] = ad_model_info
    trids = set()
    for shortform in shortforms:
        trids.update(get_text_ref_ids_for_agent_text(shortform))
    trids = list(trids)
    texts = list(get_plaintexts_for_text_ref_ids(trids, contains=shortforms))
    disambiguations = disamb.disambiguate(texts)

    dp_idx_dict = defaultdict(list)
    no_dp_idx_dict = defaultdict(list)
    for i, res in enumerate(disambiguations):
        grounding = res["decision"]
        defining_patterns = res["defining_patterns"]
        from_defining_pattern = defining_patterns is not None and len(defining_patterns) == 1
        if from_defining_pattern:
            dp_idx_dict[grounding].append(i)
        else:
            no_dp_idx_dict[grounding].append(i)
    intervals = {}
    frequencies = {}
    for grounding, model_info in ad_models.items():
        if model_info is None:
            intervals[grounding] = None
            frequencies[grounding] = None
            continue
        shape_params = model_info["shape_params"]
        sens_a, sens_b = shape_params["sens_alpha"], shape_params["sens_beta"]
        spec_a, spec_b = shape_params["spec_alpha"], shape_params["spec_beta"]
        model = GroundingAnomalyDetector.load_model_info(model_info["model"])
        idx = no_dp_idx_dict[grounding]
        with_dp_idx = dp_idx_dict[grounding]
        preds = model.predict([texts[i] for i in idx])
        mask = np.ones(len(texts), dtype=bool)
        mask[idx] = False
        mask[with_dp_idx] = False
        out_preds = model.predict([texts[i] for i, cond in enumerate(mask) if cond])
        n = len(preds)
        t = np.sum(preds == -1.0)
        m = len(out_preds)
        u = np.sum(out_preds == -1.0)
        frequencies[grounding] = [n, int(t), m, int(u)]
        metrics = sample_estimated_metrics(
            n, t, m, u, sens_a, sens_b, spec_a, spec_b, n_samples=10000, rng=1729
        )
        met = metrics["standard"]
        precision_interval = _hdi_from_sample(met.precision, alpha=0.9)
        recall_interval = _hdi_from_sample(met.recall, alpha=0.9)

        met = metrics["consensus"]
        precision_interval_consensus = _hdi_from_sample(met.precision, alpha=0.9)
        recall_interval_consensus = _hdi_from_sample(met.recall, alpha=0.9)
        intervals[grounding] = {
            "standard": {"precision": precision_interval, "recall": recall_interval},
            "consensus": {
                "precision": precision_interval_consensus,
                "recall": recall_interval_consensus
            }
        }
    return {
        "intervals": intervals,
        "ad_models": ad_models,
        "frequencies": frequencies,
        "grounding_dict": grounding_dict,
        "names": names,
        "adeft_model": adeft_model.get_model_info(),
    }
