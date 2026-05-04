import argparse
from itertools import chain
import logging
import pickle
from multiprocessing import Pool


from adeft import available_shortforms
from adeft.disambiguate import load_disambiguator


from adeft_indra.training import adeft_trainer


logger = logging.getLogger(__file__)

models = []
seen = set()
for shortform, model_name in available_shortforms.items():
    if model_name in seen:
        continue
    models.append(load_disambiguator(shortform))
    seen.add(model_name)


def get_test_cases_for_model(disamb):
    return adeft_trainer.get_opaque_test_cases_from_adeft_model(disamb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outpath')
    parser.add_argument('--n_jobs', type=int, default=1)
    args = parser.parse_args()
    outpath = args.outpath
    n_jobs = args.n_jobs
    with Pool(n_jobs) as pool:
        result = pool.map(get_test_cases_for_model, models)
    result = list(chain(*result))
    with open(outpath, 'wb') as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
