import argparse
import json

from adeft_indra.results import ResultsManager

from adeft_indra.auto_adeft import auto_adeft

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_results_db_path")

    args = parser.parse_args()
    input_results_manager = ResultsManager(args.input_results_db_path)

    for model_name, training_info in input_results_manager.items():
        with open(f"/home/birbir/extra_storage/new_models/{model_name}.json", "w") as f:
            json.dump(training_info, f, indent=True)
        
