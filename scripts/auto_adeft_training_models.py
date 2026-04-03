import argparse
import json
import logging

from glob import glob
from multiprocessing import Pool
from pathlib import Path

from adeft_indra.results import ResultsManager

from adeft_indra.auto_adeft import auto_adeft
from adeft_indra.cluster_longforms import get_names_and_pos_labels


logger = logging.getLogger(__file__)



def process_case(model_name, shortforms, grounding_dict, names, pos_labels, results_db):
    results_db = ResultsManager(results_db)
    if model_name in results_db:
        logger.info(f"Model {model_name} already trained, skipping")
        return
    logger.info(f"Attempting to train model {model_name}")
    try:
        results = auto_adeft(shortforms, grounding_dict, names, pos_labels)
    except Exception as e:
        logger.warning(f"Failure for model {model_name}")
        results_db[model_name] = {"Failed": True, "exception": e}
        return

    results_db[model_name] = results
    logger.info(f"Success for model {model_name}")


if __name__ == "__main__":
    input_db = ResultsManager("/home/birbir/extra_storage/auto_adeft_training_fixed_up4.db")

    cases = [
        [
            model_name,
            training_info["shortforms"],
            training_info["grounding_dict"],
            training_info["names"],
            training_info["pos_labels"],
            "/home/birbir/extra_storage/auto_adeft_models_3.db",
        ]
        for model_name, training_info in input_db.items()
    ]

    with Pool(8) as pool:
        pool.starmap(process_case, cases)
