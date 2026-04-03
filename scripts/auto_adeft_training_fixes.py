import argparse
import json

from glob import glob
from pathlib import Path

from adeft_indra.results import ResultsManager

from adeft_indra.auto_adeft import auto_adeft
from adeft_indra.cluster_longforms import get_names_and_pos_labels




if __name__ == "__main__":
    results = ResultsManager("/home/birbir/extra_storage/auto_adeft_training_fixed_up4.db")
    for path in glob("/home/birbir/extra_storage/new_models/*.json"):
        path = Path(path)
        with open(path) as f:
            try:
                model_info = json.load(f)
            except Exception as e:
                print("************************")
                print(path)
                raise(e)
        shortforms = model_info["shortforms"]
        grounding_dict = model_info["grounding_dict"]
        names, pos_labels = get_names_and_pos_labels(grounding_dict)
        model_name = path.stem
        results[model_name] = {
            "shortforms": shortforms,
            "grounding_dict": grounding_dict,
            "names": names,
            "pos_labels": pos_labels,
        }
    
