import argparse
from multiprocessing import Pool, Lock
import numpy as np
import pickle
import random
from typing import List, Optional, Tuple


from indra_db_lite import get_plaintexts_for_text_ref_ids
from opaque.nlp.models import GroundingAnomalyDetector
from opaque.train import train_anomaly_detector

from adeft_indra.anomaly_detection.results import ResultsManager


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
        train_trids,
        test_data,
        nu_list,
        max_features_list,
        run_name,
        predict_shape_params,
    ) = args
    with lock:
        print(
            "Started: "
            f"{model_name}--{agent_texts}--{curie}--{nu_list}--"
            f"{max_features_list}"
        )
    train_texts = list(get_plaintexts_for_text_ref_ids(train_trids))
    # Exclude texts that appear in the training data from the test
    # data.
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
        predict_shape_params=predict_shape_params,
    )
    ad_model = GroundingAnomalyDetector.load_model_info(result["model"])

    test_data = [
        (text, test_data[trid], trid)
        for trid, text in test_texts.trid_content_pairs()
    ]
    if test_data:
        test_texts, test_labels, test_trids = zip(*test_data)
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
    }
    key = get_key(model_name, curie, nu_list, max_features_list)
    ResultsManager.insert(run_name, key, result)
    with lock:
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
    parser.add_argument('run_name')
    parser.add_argument('--nu_list', nargs='+', type=float)
    parser.add_argument('--mf_list', nargs='+', type=int)
    parser.add_argument('--n_jobs', type=int, default=1)
    parser.add_argument('--predict_shape_params', action='store_true')
    lock = Lock()
    args = parser.parse_args()
    test_cases_path = args.test_cases_path
    with open(test_cases_path, 'rb') as f:
        test_cases = pickle.load(f)
    run_name = args.run_name
    nu_list = args.nu_list
    mf_list = args.mf_list
    n_jobs = args.n_jobs
    predict_shape_params = args.predict_shape_params
    if run_name not in ResultsManager.show_tables():
        ResultsManager.add_table(run_name)
    test_cases = [
        case + (
            nu_list,
            mf_list,
            run_name,
            predict_shape_params,
        )
        for case in test_cases
        if (
                ResultsManager.get(
                    run_name, get_key(case[0], case[2], nu_list, mf_list)
                )
                is None
        )
    ]

    gen = random.Random(1729)
    gen.shuffle(test_cases)
    with Pool(n_jobs) as pool:
        pool.map(
            process_test_case, test_cases, chunksize=1
        )
