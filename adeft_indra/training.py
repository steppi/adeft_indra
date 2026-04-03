import gzip
import json
import logging

from collections import Counter
from pathlib import Path

import adeft
import gilda

from adeft.discover import AdeftMiner
from adeft.locations import ADEFT_PATH
from adeft.modeling.classify import AdeftClassifier
from adeft.modeling.label import AdeftLabeler

from indra.ontology.bio import BioOntology

from indra_db_lite.api import get_text_ref_ids_for_agent_text
from indra_db_lite.api import get_plaintexts_for_text_ref_ids


bio_ont = BioOntology()


logger = logging.getLogger(__file__)


def adeftify(shortforms, *, cutoff=2.0):
    """Identify longform expansions for shortforms in texts from indra_db_lite

    Parameters
    ----------
    shortforms : list[str]
    cutoff : Optional[float]
        Adeft's Acromine based algorithm scores each potential longform
        expansion. Filter out all candidate longform expansions with score
        below `cutoff`. Default: 2.0

    Returns
    -------
    dict[str, list[tuple[str, int, float]]]
        A dictionary mapping shortforms to lists of tuples. Each tuple
        has three entries, a proposed longform expansion, the count of the
        number of times this longform expansion appeared in the text
        corpus that was pulled from ``indra_db_lite``, and the score
        assigned to this longform expansion by Adeft's scoring algorithm.

    """
    miners = {}
    for shortform in shortforms:
        trids = get_text_ref_ids_for_agent_text(shortform)
        content = get_plaintexts_for_text_ref_ids(trids, contains=shortforms)
        miners[shortform] = AdeftMiner(shortform)
        miners[shortform].process_texts(content)
        del content

    longforms_dict = {}
    for shortform in shortforms:
        longforms = miners[shortform].get_longforms()
        longforms = [
            (longform, count, score) for longform, count, score in longforms
            if count*score > cutoff
        ]
        longforms_dict[shortform] = longforms
    return longforms_dict


def comp_key(gilda_match):
    namespace = gilda_match.get_namespaces().pop()
    namespace_priority = {"FPLX": 0, "HGNC": 1}.get(namespace, 2)
    return (namespace_priority, gilda_match.score)


top_level_mesh_terms_of_interest = {
    "D007287", # Inorganic Chemicals
    "D009930", # Organic Chemicals
    "D006571", # Heterocyclic Compounds
    "D011083", # Polycyclic Compounds
    "D046911", # Macromolecular Substances
    "D006730", # Hormones, Hormone Substitutes, and Hormone Antagonists
    "D045762", # Enzymes and Coenzymes
    "D002241", # Carbohydrates
    "D008055", # Lipids
    "D000602", # Amino Acids, Peptides, and Proteins
    "D009706", # Nucleic Acids, Nucleotides, and Nucleosides
    "D045424", # Complex Mixtures
    "D001685", # Biological Factors
    "D001697", # Biomedical and Dental Materials
    "D004364", # Pharmaceutical Preparations
    "D020164", # Chemical Actions and Uses
    "D007239", # Infections
    "D009369", # Neoplasms
    "D009140", # Musculoskeletal Diseases
    "D004066", # Digestive System Diseases
    "D009057", # Stomatognathic Diseases
    "D012140", # Respiratory Tract Diseases
    "D010038", # Otorhinolaryngologic Diseases
    "D009422", # Nervous System Diseases
    "D005128", # Eye Diseases
    "D000091642", # Urogenital Diseases
    "D002318", # Cardiovascular Diseases
    "D006425", # Hemic and Lymphatic Diseases
    "D009358", # Congenital, Hereditary, and Neonatal Diseases
    "D017437", # Skin and Connective Tissue Diseases
    "D009750", # Nutritional and Metabolic Diseases
    "D004700", # Endocrine System Diseases
    "D007154", # Immune System Diseases
    "D007280", # Disorders of Environmental Origin
    "D000820", # Animal Diseases
    "D013568", # Pathological Conditions, Signs and Symptoms
    "D009784", # Occupational Diseases
    "D064419", # Chemically-Induced Disorders
    "D014947", # Wounds and Injuries
}


def mesh_term_of_interest(mesh_id):
    descendants = bio_ont.descendants_rel("MESH", mesh_id, ["isa"])
    descendants = {id_ for _, id_ in descendants}
    return bool(descendants & top_level_mesh_terms_of_interest)


def auto_ground_longforms(longforms_dict):
    """Try to ground proposed longforms with Gilda.

    ``gilda.ground`` can find multiple potential groundings. This just
    chooses the top grounding.

    Parameters
    ----------
    longforms_dict : dict[str, list[tuple, int, float]]
        A ``longforms_dict`` as produced by `adeftify`.

    Returns
    -------
    grounding_dict : dict[str, dict[str, str]]
        A dictionary mapping shortforms to inner dictionaries
        which themselves map longform expansions to the top groundings
        found with gilda. Groundings are in the form ``f"{db}:{id}"``
        (e.g. HGNC:6091, GO:GO:0072593). Longform expansions for which
        Gilda couldn't find a grounding are mapped to ``"ungrounded"``
        in the inner dictionaries. It is assumed that each shortform
        is equivalent in the sense that they should have essentially the
        same set of possible groundings. One example is "NP" and "NPs",
        where the later is the plural form of the former.
    names : dict[str, str]
        A dictionary mapping groundings of the form ``f{db}:{id}`` to
        canonical names.  Names for all top groundings found by Gilda for
        the input longform expansions across all shortforms are included.
    pos_labels : list[str]
        A list of groundings corresponding to positive labels.
        The intention is that statements with agents grounded to anything
        which isn't a positive label should be filtered out entirely.
        Positive labels correspond to groundings which are of interest
        within INDRA statements. Currently, all top groundings Gilda
        finds for the input longform expansions are considered positive
        labels. 

    Notes
    -----
  
    `auto_ground_longforms` is experimental and currently its output should be
    reviewed by a human annotator.
    
    """
    grounding_dict = {}
    names = {}
    for shortform, longforms in longforms_dict.items():
        grounding_map = {}
        for longform, _, _ in longforms:
            # if the longform is the shortform itself, then filter it out. We haven't
            # disambiguated anything.
            if longform == shortform.lower() or len(longform) < len(shortform):
                continue
            groundings = sorted(gilda.ground(longform), key=comp_key)
            if groundings:
                grounding_term = groundings[0].term
                grounding = f"{grounding_term.db}:{grounding_term.id}"
                grounding_map[longform] = grounding
                names[grounding] = groundings[0].term.entry_name
            else:
                grounding_map[longform] = "ungrounded"
        grounding_dict[shortform] = grounding_map
    candidate_pos_labels = list(names.keys())
    # filter out mesh labels for entities deemed not of interest from pos_labels
    pos_labels = []
    for label in candidate_pos_labels:
        if label.startswith("MESH:"):
            _, mesh_id = label.split(":", maxsplit=1)
            if not mesh_term_of_interest(mesh_id):
                continue
        pos_labels.append(label)
        
    return grounding_dict, names, pos_labels


def build_corpus(grounding_dict):
    """Build a corpus for model training based on a grounding dictionary.

    Parameters
    ----------
    grounding_dict : dict[str, dict[str, str]]
        A grounding dictionary in the form returned by `auto_ground_longforms`.
        Again, it is assumed that each shortform that appears as a key of the
        outer dictionary is equivalent in the sense that they should have
        essentially the same set of possible groundings. One example is "NP"
        and "NPs", where the later is the plural form of the former.

    Returns
    -------
    corpus : list[tuple[str, str]]
        A list of tuples. Each tuple contains three elements, a text document,
        an associated label for the text document. 
    """
    shortforms = list(grounding_dict.keys())
    labeler = AdeftLabeler(grounding_dict)
    corpus = []
    seen_trids = set()
    for shortform in grounding_dict.keys():
        trids = get_text_ref_ids_for_agent_text(shortform)
        trids = set(trids) - seen_trids
        seen_trids.update(trids)
        content = get_plaintexts_for_text_ref_ids(trids, contains=shortforms)
        corpus.extend(
            labeler.build_from_texts(
                (text, trid) for trid, text in content.trid_content_pairs()
            )
        )
    return corpus


def get_existing_grounding_info(shortform, *, path=ADEFT_PATH):
    """Get grounding_map, names, and pos_labels for an existing adeft model.

    Parameters
    ----------
    shortform : str
        Look up the model for this shortform. For models with multiple
        shortforms, one only needs to pick one of them.

    path : Optional[str]
        By default, `get_existing_grounding_map` uses the models for the
        installed version of Adeft, but one may optionally specify a path
        to the folder for a different Adeft version if one wants to get
        the grounding info for a past model.

    """
    path = Path(path)
    path /= "models"
    available = adeft.get_available_models(path=path)
    model_name = available[shortform]
    model_path = path / model_name
    with open(model_path / f"{model_name}_grounding_dict.json") as f:
        grounding_map = json.load(f)
    with open(model_path / f"{model_name}_names.json") as f:
        names = json.load(f)
    with gzip.GzipFile(model_path / f"{model_name}_model.gz") as f:
        json_bytes = f.read()
    model_info = json.loads(json_bytes.decode('utf-8'))
    pos_labels = model_info["pos_labels"]
    return grounding_map, names, pos_labels


def validate_and_refit_model(
        shortforms,
        corpus,
        pos_labels,
        *,
        cv=5,
        parameters=None,
        random_state=None,
        n_jobs=1,
        min_class_size=10,
):
    """Build a corpus and then validate and train a model."""
    if parameters is None:
        parameters = {
            "C": 100.0, "ngram_range": (1, 2), "max_features": 10000,
            "class_weight": "balanced"
        }

    # AdeftClassifier uses GridSearchCV, so we need to turn our parameters into
    # a param_grid even though no grid search is done here. A grid search would
    # result in data leakage since we use all data here and have no untouched
    # hold out set. The idea is just to pick a reasonable set of parameters and
    # use it everywhere without parameter tuning. Since we're using logistic
    # regression with very simple features, we can get away with this.  To
    # explore more flexible models or do any kind of model comparison we
    # will need a proper validation pipeline.
    param_grid = {key: [val] for key, val in parameters.items()}
    model = AdeftClassifier(shortforms, pos_labels, random_state=random_state)
    X, y, trids = zip(*corpus)
    counts = Counter(y)
    keep = [
        (text, label, trid) for text, label, trid in zip(X, y, trids)
        if counts[label] >= min_class_size
    ]
    if not keep:
        logger.warning(
            "No data remains after excluding classes with fewer than"
            f" {min_class_size} examples. Returning None."
        )
        return None
    X, y, trids = zip(*keep)
    if len(set(y)) == 1:
        logger.warning(
            "Only a single class remains after excluding classes with fewer"
            f" than {min_class_size} examples. Returning None."
        )
        return None
    model.cv(X, y, param_grid=param_grid, n_jobs=n_jobs, cv=cv)
    return model
