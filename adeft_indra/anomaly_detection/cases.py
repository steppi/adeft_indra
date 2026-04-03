from indra.databases.hgnc_client import get_uniprot_id
from indra.literature.pubmed_client import get_ids_for_mesh

from indra_db_lite import get_entrez_pmids_for_hgnc
from indra_db_lite import get_entrez_pmids_for_uniprot
from indra_db_lite import get_mesh_terms_for_grounding
from indra_db_lite import get_plaintexts_for_text_ref_ids
from indra_db_lite import get_pmids_for_mesh_term
from indra_db_lite import get_text_ref_ids_for_pmids

from opaque.nlp.featurize import BaselineTfidfVectorizer


def get_training_cases_for_grounding(
        namespace, identifier, *, exclude_trids=None
):
    if exclude_trids is None:
        exclude_trids = set()
    exclude_trids = set(exclude_trids)

    entrez_pmids = set()
    mesh_pmids = set()
    mesh_terms = None
    if namespace == 'HGNC':
        entrez_pmids.update(get_entrez_pmids_for_hgnc(identifier))
        uniprot_id = get_uniprot_id(identifier)
        mesh_terms = get_mesh_terms_for_grounding(namespace, identifier)
        if not mesh_terms:
            mesh_terms = get_mesh_terms_for_grounding('UP', uniprot_id)
    elif namespace == 'UP':
        entrez_pmids.update(get_entrez_pmids_for_uniprot(identifier))
        mesh_terms = get_mesh_terms_for_grounding(namespace, identifier)
        if mesh_terms:
            for mesh_id in mesh_terms:
                mesh_pmids.update(
                    [
                        int(id_)
                        for id_ in get_ids_for_mesh(mesh_id, major_topic=True)
                    ]
                )
    elif namespace == 'MESH':
        mesh_terms = [identifier]
    else:
        mesh_terms = get_mesh_terms_for_grounding(namespace, identifier)

    if mesh_terms:
        for mesh_id in mesh_terms:
            mesh_pmids.update(
                get_pmids_for_mesh_term(mesh_id, major_topic=True)
            )
    pmids = list(entrez_pmids | mesh_pmids)
    if not pmids:
        return None

    entrez_trids = set(
        get_text_ref_ids_for_pmids(entrez_pmids).values()
    )
    mesh_trids = set(
        get_text_ref_ids_for_pmids(mesh_pmids).values()
    )
    trids = list(entrez_trids | mesh_trids)
    train_data = get_plaintexts_for_text_ref_ids(
        trids, text_types=['fulltext', 'abstract']
    )
    train_data = [
        (trid, text) for trid, text in train_data.trid_content_pairs()
        if len(text) > 5 and
        not {'xml', 'elsevier', 'doi', 'article'} <=
        set(BaselineTfidfVectorizer()._preprocess(text))
    ]
    if len(train_data) < 5:
        return None
    train_trids, train_texts = zip(*train_data)
    train_data = None
    train_trids_set = set(train_trids)
    entrez_trids = [
        trid for trid in entrez_trids if trid in train_trids_set
        if trid not in exclude_trids
    ]
    mesh_trids = [
        trid for trid in mesh_trids if trid in train_trids_set
        if trid not in exclude_trids
    ]
    return {
        "mesh_terms": mesh_terms,
        "num_entrez": len(entrez_trids),
        "num_mesh": len(mesh_trids),
        "train_trids": train_trids,
    }
