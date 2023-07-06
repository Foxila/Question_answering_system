import os

WEAVIATE_PORT = os.getenv('WEAVIATE_PORT', '8080')
WEAVIATE_HOST = os.getenv('WEAVIATE_HOST', 'http://localhost')
CLASSIFIER_URL = os.getenv('CLASSIFIER_URL', "http://localhost:5000")
PATENTS_PATH_GLOB = "../../data/patent_dataset/with/US*"
USE_GPU = os.getenv('USE_GPU', "False").lower() in ['true']
RESTORE_FROM_BACKUP = os.getenv('RESTORE_FROM_BACKUP', "true").lower() in ['true']
BACKUP_NAME = 'cpc'
SAMPLE_LIMIT = int(os.getenv('SAMPLE_LIMIT', 0))
READER_MODEL = os.getenv("READER_MODEL", "deepset/roberta-base-squad2")
RETRIEVER_MODEL = os.getenv("RETRIEVER_MODEL", "sentence-transformers/multi-qa-mpnet-base-dot-v1")
SPLIT_ROUTING = os.getenv('SPLIT_ROUTING', "False").lower() in ['true']
