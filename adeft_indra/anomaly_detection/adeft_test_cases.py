import argparse
import logging
from multiprocessing import Pool


from adeft import available_shortforms
from adeft.disambiguate import load_disambiguator


from adeft_indra.training import adeft_trainer

from adeft_indra.results import ResultsManager


logger = logging.getLogger(__file__)


def get_test_cases_for_model(arg):
    model_name, disamb, results_db_path = arg
    results_db = ResultsManager(results_db_path)
    print(f"Generating test cases for {model_name}")
    cases = adeft_trainer.get_opaque_test_cases_from_adeft_model(disamb)
    print(f"Success for {model_name}")
    results_db[model_name] = cases


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath')
    parser.add_argument('--n_jobs', type=int, default=1)

    args = parser.parse_args()
    outpath = args.outpath
    n_jobs = args.n_jobs

    inputs = []
    seen = set()

    results_db = ResultsManager(outpath)
    for shortform, model_name in available_shortforms.items():
        if model_name in seen:
            continue
        seen.add(model_name)
        if model_name in results_db:
            print(f"Results already computed for {model_name}")
            continue
        inputs.append([model_name, load_disambiguator(shortform), outpath])
        seen.add(model_name)

    with Pool(n_jobs) as pool:
        for _ in pool.imap_unordered(get_test_cases_for_model, inputs):
            pass
