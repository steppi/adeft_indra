import json
import logging

from pathlib import Path

import torch
import gilda

from sentence_transformers import SentenceTransformer

from indra.databases.hgnc_client import get_uniprot_id
from indra.literature.pubmed_client import get_ids_for_mesh
from indra.ontology.bio import BioOntology

from indra_db_lite import get_entrez_pmids_for_hgnc
from indra_db_lite import get_entrez_pmids_for_uniprot
from indra_db_lite import (
    get_mesh_terms_for_grounding as _get_mesh_terms_for_grounding
)
from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_text_ref_ids_for_agent_text
from indra_db_lite import get_text_ref_ids_for_pmids
from indra_db_lite import (
    get_text_ref_ids_sources_and_agent_texts_for_grounding
)

from opaque.nlp.featurize import BaselineTfidfVectorizer

from adeft.construct import (
    AdeftConstructor, AdeftTrainer, GroundingClusterer,
    DistantEvalCorpusConstructor
)
from adeft.util import get_canonical_model_name

from adeft_indra.results import ResultsManager


logger = logging.getLogger(__file__)


bio_ont = BioOntology()


class ContentWrapper:
    """Shim for `indra_db_lite.api.TextContent`

    Bridge the API of `TextContent` with what is expected by
    `AdeftConstructor`
    """

    def __init__(self, text_content):
        self.text_content = text_content

    def items(self):
        return self.text_content.trid_content_pairs()

    def values(self):
        return iter(self.text_content)


get_content_ids_for_agent_text = get_text_ref_ids_for_agent_text


def _is_db_source(source):
    # Although these aren't the only db sources, they are the only ones
    # that currently include evidence backed by articles, so they are the
    # only ones that will appear in get_counts_for_grounding.
    return source in {"phosphoelm", "ttrust"}


def get_counts_for_grounding(curie):
    """Get counts of articles in indra_db with mention of grounding."""
    res = get_text_ref_ids_sources_and_agent_texts_for_grounding(curie)
    db_ids = set()
    reader_ids = set()
    for source, (id_, _) in res.items():
        if _is_db_source(source):
            db_ids.add(id_)
        else:
            reader_ids.add(id_)
    return len(db_ids), len(reader_ids)


def get_plaintexts_for_content_ids(ids, *, contains=None):
    return ContentWrapper(
        get_plaintexts_for_text_ref_ids(ids, contains=contains)
    )


def _comp_key(gilda_match):
    namespace = gilda_match.get_namespaces().pop()
    namespace_priority = {"FPLX": 0, "HGNC": 1}.get(namespace, 2)
    return (namespace_priority, gilda_match.score)


def ground(agent_text):
    groundings = sorted(gilda.ground(agent_text), key=_comp_key)
    if not groundings:
        return "ungrounded"
    term = groundings[0].term
    return f"{term.db}:{term}"


def get_name(grounding):
    db, id_ = grounding.split(":", maxsplit=1)
    names = gilda.get_names(grounding, id_, status="name")
    if not names:
        return ""
    return names[0]


top_level_mesh_terms_of_interest = {
    "D007287",  # Inorganic Chemicals
    "D009930",  # Organic Chemicals
    "D006571",  # Heterocyclic Compounds
    "D011083",  # Polycyclic Compounds
    "D046911",  # Macromolecular Substances
    "D006730",  # Hormones, Hormone Substitutes, and Hormone Antagonists
    "D045762",  # Enzymes and Coenzymes
    "D002241",  # Carbohydrates
    "D008055",  # Lipids
    "D000602",  # Amino Acids, Peptides, and Proteins
    "D009706",  # Nucleic Acids, Nucleotides, and Nucleosides
    "D045424",  # Complex Mixtures
    "D001685",  # Biological Factors
    "D001697",  # Biomedical and Dental Materials
    "D004364",  # Pharmaceutical Preparations
    "D020164",  # Chemical Actions and Uses
    "D007239",  # Infections
    "D009369",  # Neoplasms
    "D009140",  # Musculoskeletal Diseases
    "D004066",  # Digestive System Diseases
    "D009057",  # Stomatognathic Diseases
    "D012140",  # Respiratory Tract Diseases
    "D010038",  # Otorhinolaryngologic Diseases
    "D009422",  # Nervous System Diseases
    "D005128",  # Eye Diseases
    "D000091642",  # Urogenital Diseases
    "D002318",  # Cardiovascular Diseases
    "D006425",  # Hemic and Lymphatic Diseases
    "D009358",  # Congenital, Hereditary, and Neonatal Diseases
    "D017437",  # Skin and Connective Tissue Diseases
    "D009750",  # Nutritional and Metabolic Diseases
    "D004700",  # Endocrine System Diseases
    "D007154",  # Immune System Diseases
    "D007280",  # Disorders of Environmental Origin
    "D000820",  # Animal Diseases
    "D013568",  # Pathological Conditions, Signs and Symptoms
    "D009784",  # Occupational Diseases
    "D064419",  # Chemically-Induced Disorders
    "D014947",  # Wounds and Injuries
}


def mesh_term_of_interest(mesh_id):
    descendants = bio_ont.descendants_rel("MESH", mesh_id, ["isa"])
    descendants = {id_ for _, id_ in descendants}
    return bool(descendants & top_level_mesh_terms_of_interest)


def get_mesh_terms_for_grounding(grounding):
    ns, id_ = grounding.split(":", maxsplit=1)
    return _get_mesh_terms_for_grounding(ns, id_)


def is_pos_label(grounding):
    db, id_ = grounding.split(":", maxsplit=1)
    if db == "MESH":
        return mesh_term_of_interest(id_)
    if db == "HGNC":
        return True
    mesh_terms = set(get_mesh_terms_for_grounding(grounding))
    return any([mesh_term_of_interest(mesh_id) for mesh_id in mesh_terms])


class CosineSimilarity:
    def __init__(
            self,
            *,
            model=None,
            batch_size=128,
    ):
        if model is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(
                "pritamdeka/S-Biomed-Roberta-snli-multinli-stsb",
                device=device,
            )
        self.model = model
        self.batch_size = batch_size

    def __call__(self, longforms):
        with torch.inference_mode():
            embeddings = self.model.encode(
                longforms,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        return embeddings @ embeddings.T


cosine_similarity = CosineSimilarity()


def nearest_common_ancestor(grounding1, grounding2):
    ns1, id1 = grounding1.split(":", maxsplit=1)
    ns2, id2 = grounding2.split(":", maxsplit=1)

    # the bio_ontology uses ancestor and descendant in the reverse order from
    # what the adeft construction api expects. For BioOntology, the more
    # general term is a descendant, for adeft, the more general term is an
    # ancestor.
    anc1 = bio_ont.descendants_rel(ns1, id1, ["isa"])
    anc2 = bio_ont.descendants_rel(ns2, id2, ["isa"])

    # The logic here is not solid enough for general production
    # use. `descendants_rel` does a breadth first search through the tree, so
    # this will pick a common ancestor of minimum height. Because the MESH tree
    # isn't actually a tree, the common ancestor may not be unique, and
    # sometimes this may pick a common ancestor that is undesirably vague over
    # another ancestor of equal height which is suitably specific. Perhaps it
    # may be useful to add weights to ["isa"] edges quantifying conceptual
    # distance in some way and use a dykstra like algorithm to find the closest
    # common ancestor.

    # For now I'm leaving this function out of INDRA since it is probably not
    # robust enough for general use. For Adeft, if an unreasonably vague
    # ancestor is chosen, the `is_pos_label` logic should be able to at least
    # cause this grounding to be an ignored negative label.
    anc_set2 = set(anc2)
    for node in reversed(anc1):
        if node in anc_set2:
            ns, id_ = node
            return f"{ns}:{id_}"
    return None


grounding_clusterer = GroundingClusterer(
    cosine_similarity, nearest_common_ancestor
)


adeft_constructor = AdeftConstructor(
    get_content_ids_for_agent_text,
    get_plaintexts_for_content_ids,
    ground,
    get_name,
    is_pos_label,
    grounding_clusterer,
    filter_func=filter_func,
)


def get_content_ids_for_gene_or_protein(grounding):
    ns, id_ = grounding.split(":", maxsplit=1)
    if ns not in {"HGNC", "UP"}:
        return []
    pmids = set()
    if ns == "HGNC":
        pmids.update(get_entrez_pmids_for_hgnc(id_))
        uniprot_id = get_uniprot_id(id_)
        pmids.update(get_entrez_pmids_for_uniprot(uniprot_id))
    elif ns == "UP":
        pmids.update(get_entrez_pmids_for_uniprot(id_))
    return list(get_text_ref_ids_for_pmids(pmids).values())


def get_content_ids_from_mesh(grounding):
    ns, id_ = grounding.split(":", maxsplit=1)
    if ns == "HGNC":
        uniprot_id = get_uniprot_id(id_)
        mesh_terms = get_mesh_terms_for_grounding("UP", uniprot_id)
    elif ns != "MESH":
        mesh_terms = get_mesh_terms_for_grounding(ns, id_)
    else:
        mesh_terms = [id_]
    pmids = set()
    for mesh_id in mesh_terms:
        pmids.update(
            (id_ for id_ in get_ids_for_mesh(mesh_id, major_topic=True))
        )
    return list(get_text_ref_ids_for_pmids(pmids).values())


def filter_func(text):
    return (
        len(text) > 5
        and not {"xml", "elsevier", "doi", "article"}
        <= set(BaselineTfidfVectorizer()._preprocess(text))
    )


disteval_constructor = DistantEvalCorpusConstructor(
    get_content_ids_for_gene_or_protein,
    get_content_ids_from_mesh,
    get_mesh_terms_for_grounding,
    get_plaintexts_for_content_ids,
    get_counts_for_grounding,
    filter_func=filter_func,
)


adeft_trainer = AdeftTrainer(adeft_constructor, disteval_constructor)


def generate_initial_candidate_grounding_info(shortforms_list, results_db):
    for shortforms in shortforms_list:
        model_name = get_canonical_model_name(shortforms)
        if model_name in results_db:
            logger.info(
                f"Grounding info already generated for shortforms {shortforms}"
            )
        logger.info(f"Generating grounding info for shortforms {shortforms}.")
        try:
            grounding_dict, names, pos_labels = (
                adeft_constructor.get_grounding_info(shortforms)
            )
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
        results_db[model_name] = result


def dump_grounding_results_to_json(results_db, outpath):
    base_path = Path(outpath).expanduser().resolve()
    base_path.mkdir(parents=True, exist_ok=True)
    for model_name, grounding_info in results_db.items():
        with open(base_path / "model_name.json", "w") as f:
            json.dump(grounding_info, f, indent=True)


def refresh_groundings_from_json(results_db, inpath):
    base_path = Path(inpath).expanduser().resolve()
    if not base_path.isdir():
        return
    model_name = base_path.stem
    for file_path in base_path.glob("*.json"):
        with open(file_path) as f:
            try:
                model_info = json.load(f)
            except Exception as e:
                logger.warning(
                    f"Failure to decode json at {file_path} due to {e}"
                )
                results_db[model_name] = {"failed": True, "exception": str(e)}
        shortforms = model_info["shortforms"]
        grounding_dict = model_info["grounding_dict"]
        names, pos_labels = adeft_constructor.get_names_and_pos_labels(
            grounding_dict
        )

        results_db[model_name] = {
            "shortforms": shortforms,
            "grounding_dict": grounding_dict,
            "names": names,
            "pos_labels": pos_labels,
        }


# def _train_model(
#         model_name, shortforms, grounding_dict, names, pos_labels, results_db_path
# ):
#     results_db = ResultsManager(results_db_path)
#     if model_name in results_db:
#         logger.info(f"Model {model_name} already trained, skipping")
#         return
#     logger.info(f"Attempting to train model {model_name}")
#     try:
#         results = auto_adeft(shortforms, grounding_dict, names, pos_labels)
#     except Exception as e:
#         logger.warning(f"Failure for model {model_name}")
#         results_db[model_name] = {"Failed": True, "exception": e}
#         return

#     results_db[model_name] = results
#     logger.info(f"Success for model {model_name}")



# def train_models(input_db, results_db_path):
#     cases = [
#         [
#             model_name,
#             training_info["shortforms"],
#             training_info["grounding_dict"],
#             training_info["names"],
#             training_info["pos_labels"],
#             results_db_path,
#         ]
#         for model_name, training_info in input_db.items()
#     ]
#     with Pool(8) as pool:
#         pool.starmap(process_case, cases)
