import argparse
from itertools import chain
import logging
import pickle
from multiprocessing import Pool


from adeft import available_shortforms
from adeft.disambiguate import load_disambiguator
from adeft.modeling.label import AdeftLabeler

from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_text_ref_ids_for_agent_text

from opaque.nlp.featurize import BaselineTfidfVectorizer

from .cases import get_training_cases_for_grounding


logger = logging.getLogger(__file__)


models = {
    model_name: load_disambiguator(shortform)
    for shortform, model_name in available_shortforms.items()
}

reverse_model_map = {
    model_name: shortform for shortform, model_name
    in available_shortforms.items()
}


def get_groundings_for_disambiguator(disamb):
    result = set()
    for grounding_map in disamb.grounding_dict.values():
        for curie in grounding_map.values():
            result.add(curie)
    return list(result)


def get_test_cases_for_model(model_name):
    vectorizer = BaselineTfidfVectorizer()

    def preprocess(text):
        return vectorizer._preprocess(text)
    result = []
    disamb = models[model_name]
    shortforms = disamb.shortforms
    trids = set()
    for shortform in shortforms:
        trids.update(get_text_ref_ids_for_agent_text(shortform))
    trids = list(trids)
    content = get_plaintexts_for_text_ref_ids(
        trids, text_types=['abstract', 'fulltext']
    )
    unlabeled = [
        (trid, text) for trid, text in content.trid_content_pairs()
        if len(text) > 5 and
        not {'xml', 'elsevier', 'doi', 'article'} <=
        set(preprocess(text))
    ]
    labeler = AdeftLabeler(grounding_dict=disamb.grounding_dict)
    test_corpus = labeler.build_from_texts(
        (text, trid) for trid, text in unlabeled
    )
    test_data = {trid: label for _, label, trid in test_corpus}
    test_corpus = None
    for curie in get_groundings_for_disambiguator(disamb):
        if ':' not in curie:
            continue
        if len([label for label in test_data.values() if label == curie]) < 5:
            continue
        namespace, identifier = curie.split(':', maxsplit=1)
        train_info = get_training_cases_for_grounding(
            namespace, identifier, exclude_trids=set(test_data)
        )
        if train_info is None:
            continue
        result.append(
                (
                    model_name,
                    tuple(shortforms),
                    curie,
                    train_info["mesh_terms"],
                    train_info["num_entrez"],
                    train_info["num_mesh"],
                    train_info["train_trids"],
                    test_data,
                )
            )
    return result


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
