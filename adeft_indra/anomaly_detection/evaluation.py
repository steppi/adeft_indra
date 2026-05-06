import argparse
from multiprocessing import Pool
import numpy as np
import random
from typing import List, Optional, Tuple


from indra_db_lite import get_plaintexts_for_text_ref_ids
from opaque.nlp.models import GroundingAnomalyDetector
from opaque.train import train_anomaly_detector

from adeft_indra.results import ResultsManager


def get_key(
        model_name: str,
        curie: str,
        nu_list: List[int],
        mf_list: List[int],
        other: Optional[str] = None
) -> str:
    key = f"{model_name}:{curie}"
    key += (str(nu_list) + str(mf_list)).replace(' ', '')
    if other:
        key += other
    return key


def process_test_case(args: Tuple) -> None:
    (
        model_name,
        agent_texts,
        curie,
        mesh_terms,
        num_entrez_texts,
        num_mesh_texts,
        num_db_texts,
        num_reader_texts,
        train_trids,
        test_data,
        nu_list,
        max_features_list,
        predict_shape_params,
        results_db_path,
    ) = args
    print(
        "Started: "
        f"{model_name}--{agent_texts}--{curie}--{nu_list}--"
        f"{max_features_list}"
    )
    train_texts = list(get_plaintexts_for_text_ref_ids(train_trids))
    test_texts = get_plaintexts_for_text_ref_ids(
        test_data,
        text_types=['abstract', 'fulltext'],
    )

    result = train_anomaly_detector(
        agent_texts,
        train_texts,
        nu_list,
        max_features_list,
        random_state=1729,
        num_mesh_texts=num_mesh_texts,
        num_entrez_texts=num_entrez_texts,
        num_db_texts=num_db_texts,
        num_reader_texts=num_reader_texts,
        predict_shape_params=predict_shape_params,
    )
    ad_model = GroundingAnomalyDetector.load_model_info(result["model"])

    test_data = [
        (text, test_data[trid], trid)
        for trid, text in test_texts.trid_content_pairs()
    ]
    if test_data:
        test_texts, test_labels, _ = zip(*test_data)
        preds = ad_model.predict(test_texts).flatten()
        test_labels = np.array(test_labels)
        tn = (preds == 1.0) & (test_labels == curie)
        tp = (preds == -1.0) & (test_labels != curie)
        sens = sum(tp) / sum(test_labels != curie)
        spec = sum(tn) / sum(test_labels == curie)
        J = sens + spec - 1
    else:
        preds, test_labels, sens, spec, J = (None, ) * 5
    result['test_stats'] = {
        'sensitivity': sens, 'specifity': spec, 'J': J
    }
    result['test_info'] = {
        'labels': test_labels,
        'preds': preds,
    }
    result['train_info'] = {
        'num_entrez_texts': num_entrez_texts,
        'num_mesh_texts': num_mesh_texts,
        'num_db_texts': num_db_texts,
        'num_reader_texts': num_reader_texts,
    }

    key = get_key(model_name, curie, nu_list, max_features_list)
    results_db = ResultsManager(results_db_path)
    results_db[key] = result
    print(
        "Success: "
        f"{model_name}--{agent_texts}--{curie}--{nu_list}--"
        f"{max_features_list}"
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Train and evaluate grounding anomaly detectors on"
        " adeft test cases."
    )
    parser.add_argument('test_cases_path')
    parser.add_argument('results_db_path')
    parser.add_argument('--nu_list', nargs='+', type=float)
    parser.add_argument('--mf_list', nargs='+', type=int)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--predict_shape_params', action='store_true')
    args = parser.parse_args()

    test_cases_path = args.test_cases_path
    results_db_path = args.results_db_path
    nu_list = args.nu_list
    mf_list = args.mf_list
    n_jobs = args.n_jobs
    predict_shape_params = args.predict_shape_params

    test_cases_db = ResultsManager(test_cases_path)
    results_db = ResultsManager(results_db_path)

    test_cases = []

    for model_name, info in test_cases_db.items():
        agent_texts = info["shortforms"]
        test_data = info["test_data"]
        for curie, training_info in info["training_info"].items():
            test_cases.append(
                [
                    model_name,
                    agent_texts,
                    curie,
                    training_info["mesh_terms"],
                    training_info["num_entrez"],
                    training_info["num_mesh"],
                    training_info["db_count"],
                    training_info["reader_count"],
                    training_info["train_ids"],
                    test_data,
                    nu_list,
                    mf_list,
                    predict_shape_params,
                    results_db_path,
                ]
            )

    test_cases = [
        case for case in test_cases
        if get_key(case[0], case[2], nu_list, mf_list) not in results_db
    ]

    gen = random.Random(1729)
    gen.shuffle(test_cases)
    with Pool(n_jobs) as pool:
        pool.map(
            process_test_case, test_cases, chunksize=1
        )
