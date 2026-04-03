import argparse
import logging
import multiprocessing

import adeft

from adeft.locations import ADEFT_PATH
from adeft.util import get_canonical_model_name

from adeft_indra.results import ResultsManager
from adeft_indra.training import build_corpus
from adeft_indra.training import get_existing_grounding_info
from adeft_indra.training import validate_and_refit_model


logger = logging.getLogger(__file__)


def retrain_model_worker(shortform, results_db_path):
    results_manager = ResultsManager(results_db_path)
    grounding_dict, names, pos_labels = get_existing_grounding_info(shortform)
    model_name = get_canonical_model_name(grounding_dict.keys())

    if model_name in results_manager:
        return

    logger.info(f"Retraining {model_name}.")
    shortforms = grounding_dict.keys()
    corpus = build_corpus(grounding_dict)
    model = validate_and_refit_model(
        shortforms,
        corpus
        pos_labels,
        random_state=1729,
        min_class_size=10,
    )
    if model is None:
        logger.warning(f"Insufficient data to retrain model {model_name}.")
        results_manager[model_name] = None
        return
    logger.info(f"{model_name} finished retraining.")
    results_manager[model_name] = model.get_model_info()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_db_path")
    parser.add_argument("--n_jobs", type=int, default=1)

    args = parser.parse_args()
    results_manager = ResultsManager(args.results_db_path)
    
    models_dict = adeft.get_available_models()
    # Exclude models which have already been retrained.
    reduced_shortforms = list(
        {
            val: key for key, val in models_dict.items() if val not in results_manager
            and key != "__TEST"
        }.values()
    )

    with multiprocessing.Pool(args.n_jobs) as pool:
        pool.starmap(
            retrain_model_worker,
            [
                (shortform, args.results_db_path)
                for shortform in reduced_shortforms
            ]
        )
