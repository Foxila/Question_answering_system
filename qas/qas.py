
from haystack.nodes import EmbeddingRetriever, JoinDocuments
from haystack.nodes.retriever import BM25Retriever
from haystack.pipelines import (Pipeline)
from haystack.utils import clean_wiki_text
from haystack.utils.preprocessing import convert_files_to_docs
from haystack.document_stores import FAISSDocumentStore
from haystack.document_stores import InMemoryDocumentStore
from haystack.utils import clean_wiki_text
from haystack.utils.preprocessing import convert_files_to_docs


class QAS:
    document_store = FAISSDocumentStore(faiss_index_factory_str="Flat", similarity="dot_product", embedding_dim=768)
    kw_store = InMemoryDocumentStore(use_bm25=True)
    retriever = EmbeddingRetriever(
        document_store=document_store,
        use_gpu=True,
        embedding_model="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        model_format="sentence_transformers"
    )
    bm25_retriever = BM25Retriever(document_store=kw_store)
    rrf_pipeline = Pipeline()
    def __init__(self):
        dir_path="./cpctxt"
        for x in range(27): 
           path = dir_path + "/" + str(x + 1)
           docs = convert_files_to_docs(path, clean_func=clean_wiki_text)
           self.kw_store.write_documents(docs)
           self.document_store.write_documents(docs)
           self.document_store.update_embeddings(retriever=self.retriever,  update_existing_embeddings=False)
        self.rrf_pipeline.add_node(component= self.retriever, name="Dense Retriever", inputs=["Query"])
        self.rrf_pipeline.add_node(component= self.bm25_retriever, name="BM25 Retriever", inputs=["Query"])
        self.rrf_pipeline.add_node(component=JoinDocuments(join_mode="reciprocal_rank_fusion"),  name="JoinResults", inputs=["Dense Retriever", "BM25 Retriever"])


    @staticmethod
    def process_pipeline(self, request) -> dict:
        quer = request.query
        params = request.params
        ret = params['Retriever']
        k = ret['top_k']
        al = params['Alpha']
        share = al['sparse_share']
        alpha_pipeline = Pipeline()
        alpha_pipeline.add_node(component= self.retriever, name="Dense Retriever", inputs=["Query"])
        alpha_pipeline.add_node(component= self.bm25_retriever, name="BM25 Retriever", inputs=["Query"])
        alpha_pipeline.add_node(component=JoinDocuments(join_mode="merge", weights = [1-share, share]),  name="JoinResults", inputs=["Dense Retriever", "BM25 Retriever"])
        output = alpha_pipeline.run(query=quer, params={"Dense Retriever": {"top_k": k}, "BM25 Retriever": {"top_k": k}})
        results = output['documents']
        dok =[]
        for i in range(k):
            dok.append(results[i])
        resdic = {}
        for doc in dok:
            name = doc.meta["name"]
            cl = name[:1]
            link = "https://www.cooperativepatentclassification.org/sites/default/files/cpc/scheme/"+ cl +"/scheme-" + name[:4] + ".pdf"
            descr = "CPC subclass " + name[:4] + " available at"
            resdic[descr] = link
        return resdic
    
    @staticmethod
    def process_rrf_pipeline(request, rrf_pipeline) -> dict:
        quer = request.query
        params = request.params
        ret = params['Retriever']
        k = ret['top_k']
        output = rrf_pipeline.run(query=quer, params={"Dense Retriever": {"top_k": k}, "BM25 Retriever": {"top_k": k}})
        results = output['documents']
        dok =[]
        for i in range(k):
            dok.append(results[i])
        resdic = {}
        for doc in dok:
            name = doc.meta["name"]
            cl = name[:1]
            link = "https://www.cooperativepatentclassification.org/sites/default/files/cpc/scheme/"+ cl +"/scheme-" + name[:4] + ".pdf"
            descr = "CPC subclass " + name[:4] + " available at"
            resdic[descr] = link
        return resdic
