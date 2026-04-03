import argparse
import json
import logging
import multiprocessing

import adeft

from adeft.locations import ADEFT_PATH
from adeft.util import get_canonical_model_name

from adeft_indra.results import ResultsManager
from adeft_indra.cluster_longforms import generate_adeft_grounding_info



logger = logging.getLogger(__file__)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("results_db_path")
    parser.add_argument("shortforms_file")


    args = parser.parse_args()
    results_manager = ResultsManager(args.results_db_path)
    with open(args.shortforms_file) as f:
        print(args.shortforms_file)
        shortforms_list = json.load(f)

    for shortforms in shortforms_list:
        model_name = get_canonical_model_name(shortforms)

        logger.info(f"Generating grounding info for shortforms {shortforms}")
        try:
            grounding_dict, names, pos_labels =  generate_adeft_grounding_info(shortforms)
        except Exception as e:
            logger.warning(f"Failure for {shortforms} due to exception {e}")
            grounding_dict = None
            names = None
            pos_labels = None
            exception = str(e)
        else:
            logger.info(f"Success for shortforms {shortforms}")
            exception = None

        result = {
            "shortforms": shortforms,
            "grounding_dict": grounding_dict,
            "names": names,
            "pos_labels": pos_labels,
            "exception": exception,
        }

        results_manager[model_name] = result
