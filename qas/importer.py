import glob
import logging

import requests
from haystack.nodes import PreProcessor
from nltk.tokenize import sent_tokenize

import settings

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=False,
    split_by="word",
    split_length=500,
    split_respect_sentence_boundary=True,
    split_overlap=10
)


def restore_from_backup():
    """Restoring a weaviate backup set up in the settings"""
    import weaviate

    client = weaviate.Client('%s:%s' % (settings.WEAVIATE_HOST, settings.WEAVIATE_PORT))
    try:
        result = client.backup.restore(
            backup_id=settings.BACKUP_NAME,
            backend='filesystem',
            wait_for_completion=True,
        )
        logging.info(result)
    except weaviate.exceptions.UnexpectedStatusCodeException as e:
        logging.error(e)


def load_corpus(document_store, retriever, limit=settings.SAMPLE_LIMIT, delete_all_documents=False,
                parts=["CLAIMS", "STATE_OF_THE_ART", "ABSTRACT", "DESCRIPTION", "SUMMARY"],
                run_classification=True):
    """
    importing corpus from file system. Path is set in the settings.

    :param document_store: DocumentStore
    :param retriever: Retriever
    :param limit: optional limit of patents to import
    :param delete_all_documents: bool if database should be cleand before
    :param parts: list of patent parts to import
    :param run_classification: bool if patent sentences should be classified and indexed
    :return:
    """

    if delete_all_documents:
        try:
            document_store.delete_documents()
        except Exception as e:
            logging.error("Couldn't delete documents", e)

    document_list = sorted(glob.glob(settings.PATENTS_PATH_GLOB))
    counter = 0

    preprocessed_docs = []

    # read patent files from file system and generate documents:
    for folder in document_list:
        patent_id = folder.split("/")[-1]
        with open(folder + "/" + patent_id + ".INVENTION_TITLE", "r") as f:
            invention_title = f.read().rstrip()

        # generate list of documents
        patent_docs = []
        for p in parts:
            try:
                with open(folder + "/" + patent_id + "." + p, "r") as f:
                    content = f.read().rstrip()
                    document = {
                        'content': content,
                        'meta': {'name': invention_title,
                                 'patent_id': patent_id,
                                 'part': p,
                                 'type': "patent"}
                    }
            except FileNotFoundError:
                logging.info("File not found: %s (%s)" % (p, patent_id))
            patent_docs.append(document)

            # Eventually run sentence-by-sentence classification for state of the art:
            if p == "STATE_OF_THE_ART" and run_classification:
                index_problem_solution_contradiction_docs(content, patent_id, document_store, retriever)

        # preprocess documents with preprocessor:
        preprocessed_docs = preprocessed_docs + preprocessor.process(patent_docs)

        # Batch upload if more than 250 docs in preprocessed_docs
        if len(preprocessed_docs) >= 250 or counter == limit:

            # Calculate the embedding for the documents
            embedding_list = retriever.embed_documents(preprocessed_docs)
            embedded_docs = []
            for i, emb in enumerate(embedding_list):
                doc = preprocessed_docs[i]
                doc.embedding = emb
                embedded_docs.append(doc)

            # Store documents in Database:
            document_store.write_documents(embedded_docs)
            # Resetting preprocessed docs:
            preprocessed_docs = []

        if limit:
            counter += 1
            if counter > limit:
                break


def extract_and_import_problems_solutions_contradictions(document_store, retriever, limit=settings.SAMPLE_LIMIT):
    """
    Extracting and indexing problems, solutions and contradictions from files. Path is set up in settings.
    :param document_store: DocumentStore
    :param retriever: Retriever
    :param limit: optional limit
    """

    document_list = sorted(glob.glob(settings.PATENTS_PATH_GLOB))

    counter = 0
    for folder in document_list:
        patent_id = folder.split("/")[-1]

        with open(folder + "/" + patent_id + "." + "STATE_OF_THE_ART", "r") as f:
            sota = f.read().rstrip()
            index_problem_solution_contradiction_docs(sota, patent_id, document_store, retriever)

        if limit:
            counter += 1
            if counter > limit:
                break


def index_problem_solution_contradiction_docs(content, patent_id, document_store, retriever):
    """
    Sentence-by-sentence classification and indexing for a given content
    :param content: str
    :param patent_id: str - id of the patent
    :param document_store: DocumentStore
    :param retriever: Retriever
    """
    problem_solution_docs = []
    contradiction_docs = []

    sentences = sent_tokenize(content)

    r = requests.post('%s/nps' % (settings.CLASSIFIER_URL), json={"sentences": sentences})
    nps_predictions = r.json()["classification"]

    for i, prediction in enumerate(nps_predictions):
        if prediction == 1:  # problem
            sentence = sentences[i]
            if len(sentence) > 25:
                document = {
                    'content': sentence,
                    'meta': {'name': "",
                             'patent_id': patent_id,
                             'type': "problem"}
                }
                problem_solution_docs.append(document)
        elif prediction == 2:  # solution
            sentence = sentences[i]
            if len(sentence) > 25:
                document = {
                    'content': sentence,
                    'meta': {'name': "",
                             'patent_id': patent_id,
                             'type': "solution"}
                }
                problem_solution_docs.append(document)

    r = requests.post('%s/contradiction' % (settings.CLASSIFIER_URL), json={"sentences": sentences})
    contradiction_predictions = r.json()["classification"]

    for i, prediction in enumerate(contradiction_predictions):
        if prediction == 1:  # contradiction
            sentence = sentences[i]
            if len(sentence) > 30:
                document = {
                    'content': sentences[i],
                    'meta': {'name': "",
                             'patent_id': patent_id,
                             'type': "contradiction"}
                }
                contradiction_docs.append(document)

    preprocessed_docs = []

    if len(problem_solution_docs) > 0:
        preprocessed_docs += preprocessor.process(problem_solution_docs)

    if len(contradiction_docs) > 0:
        preprocessed_docs += preprocessor.process(contradiction_docs)

    embedding_list = retriever.embed_documents(preprocessed_docs)
    embedded_docs = []
    for i, emb in enumerate(embedding_list):
        doc = preprocessed_docs[i]
        doc.embedding = emb
        embedded_docs.append(doc)
    document_store.write_documents(embedded_docs)
