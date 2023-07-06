import uvicorn
from fastapi import FastAPI
from validation import QueryRequest
from qas import QAS

app = FastAPI()
qas = QAS()

@app.get("/")
def home():
    return {"Question Answering": "System"}

if __name__ == "__main__":
    uvicorn.run("question_answering_system:app")

@app.post("/query")
def pc_index_query(request: QueryRequest):
    result = qas.process_pipeline(qas, request)
    return result


@app.post("/rrf_query")
def rrf_query(request: QueryRequest):
    result = qas.process_rrf_pipeline(request, qas.rrf_pipeline)
    return result


@app.get("/running")
def running():
    return "OK"