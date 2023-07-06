import requests
import streamlit as st
from annotated_text import annotated_text
import time
import os
import math

QAS_BASE_URL = os.getenv('QAS_BASE_URL', 'http://host.docker.internal:8000')
qas_up = False

st.set_page_config(
    page_title="Patent-Question-Answering",
    layout="centered",
    initial_sidebar_state="auto",
)

st.title('Patent QA-System')
st.subheader('Enter your question to find innovative solutions')

session = requests.Session()

# Set up Sidebar:
st.sidebar.subheader('Options')
with st.form("filter"):
    with st.sidebar:
        alpha = st.sidebar.slider('Apha for Retrieval (1 means only sparse retrieval, 0 means only dense retrieval):', min_value=0.0, max_value=1.0, value=1.0)
        nr_answers = st.sidebar.slider('Max. Answers:', min_value=1, max_value=30, value=10)

        rrf_retrieval = st.sidebar.checkbox("Reciprocal Rank Fusion", value=False,
                                             help="Reciprocal Rank Fusion (RRF) is a simple method for combining the document rankings from multiple IR systems")

        patent_class = st.sidebar.text_input("Find patents of the subclass:")
        patent_year = st.sidebar.text_input("Year:")
        filter_submitted = st.form_submit_button(label='Get a link to patents')

if filter_submitted:
    int_year = int(patent_year)
    if type(int_year) != int or int(math.log10 (int_year))+1 !=4:
        st.write("Please enter the year as integer with four digits")
    if int_year < 1971:
        st.write("Please enter a year, that is not before 1971")
    if int_year > 2023:
        st.write("Please enter a year before 2024")
    else:
       link = "https://patents.google.com/?q=" + patent_class + "&before=publication:" + str(int_year + 1) + "0101&after=publication:" + patent_year + "0101"
       st.write("The patents are available at:" + link )

# New requests should only be forwarded to the QA system when filters are updated or a new question was entered
# it should not happen automatically every time a change is made, as this is very computationally intensive.
if 'question_submitted' not in st.session_state:
    st.session_state['question_submitted'] = False


def set_question_submitted():
    st.session_state['question_submitted'] = True


# Setup Question Input and a placeholder position for answers:
question = st.text_input('', on_change=set_question_submitted)
answers_placeholder = st.empty()


def qas_query(session, data, rrf_retrieval=False):
    """Routing and sending request to QA-System"""

    if rrf_retrieval:
        url = QAS_BASE_URL + "/rrf_query"
    else:
        url = QAS_BASE_URL + "/query"

    result = session.post(url, json=data, headers={
        "Content-Type": "application/json",
        "accept": "application/json"
    })
    return result.json()


def annotate_context(answer, context):
    """Mark answers in questions with background color"""

    idx = context.find(answer)
    idx_end = idx + len(answer)
    annotated_text(context[:idx], (answer, "", "#76A5AF"), context[idx_end:], )


def running():
    """Check if QA-System is already ready"""

    url = QAS_BASE_URL + "/running"
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return True
    except:
        return False


# Show loading spinner until QA System is not running:
with st.spinner("QAS System is not running yet - Waiting."):
    while not running():
        time.sleep(3)


# Pipeline to generate http request for the QA-System, sending it and processing the response
def run_pipeline():
    with st.spinner(text='Retrieving patent documents and extracting answers...'):
        if True:
            query = {
                "query": question,
                "params": {
                    "Alpha": {"sparse_share": alpha},
                    "Retriever": {"top_k": nr_answers}
                }
            }

            # Routing query to different endpoint if option for deep retrieval is selected or not:
            if rrf_retrieval:
                response = qas_query(session, data=query, rrf_retrieval=True)
            else:
                response = qas_query(session, data=query)

            st.markdown("""---""")
            # Write Answers:
            try:
                answer = response
                for key, value in answer.items():
                    answers_placeholder.write("{}: {}".format(key, value))

            except IndexError:
                st.write("No more relevant Answers found")


# Run request pipeline if new question was entered or filter form was submitted
if question and (st.session_state['question_submitted'] == True):
    st.session_state['question_submitted'] = False
    run_pipeline()
