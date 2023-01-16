 # Copyright : Petroglyphs NLP Consulting
 # author: Jerome Massot (jerome.massot.78@gmail.com)

from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
from transformers import pipeline

from pinecone import PineconeProtocolError
import pinecone

import streamlit as st
import pandas as pd


@st.experimental_singleton
def init_pinecone():
    """
    Init the pinecone index with api key which is treated as secret
    when the app is deployed on Streamlit Cloud.
    :return: pinecone index object
    """
    api_key = st.secrets['API_KEY']
    pinecone.init(api_key=api_key, environment='us-west1-gcp')
    return pinecone.Index('video-index-merged')
    

@st.experimental_singleton
def init_retriever():
    """
    Return the LLM indicated in the code
    :return: Language Model object
    """
    return SentenceTransformer('multi-qa-mpnet-base-dot-v1')


@st.experimental_singleton
def load_qa_pipeline():
    """
    Load the HuggingFace pipeline for Question-Answering
    :return: HuggingFace pipeline object
    """
    return pipeline("question-answering", model='deepset/roberta-base-squad2')


def reconstruct_answered_context(index, query, top_k=3, origin=['Google', 'IBM']):
    """
    Find the contexts used for the Q&A engine using the query
    :param query: question asked by the user
    :param top_k: number of contexts to return from the index
    :param origin: origin of the videos to be used as filter
    :return: list of sentences as string objects
    """
 
    # embed the query
    xq = retriever.encode([query]).tolist()
    
    # search the contexts
    try:
        xc = index.query(
            xq, 
            top_k=top_k,
            filter={'origin': {"$in": origin}},
            include_metadata=True
        )
    except PineconeProtocolError:
        index = init_pinecone()
        xc = index.query(
            xq, 
            top_k=top_k,
            filter={'origin': {"$in": origin}},
            include_metadata=True
        )
    
    # reconstruct the contexts and find the answers
    returned_contexts_answers = list()
    for context in xc['matches']:
        text_context = context['metadata']['text']

        returned_contexts_answers.append({
            'title': context['metadata']['title'],
            'text': text_context,
            'origin': context['metadata']['origin'],
            'url': context['metadata']['url'], 
            'keywords': context['metadata']['keywords'],
            'start': context['metadata']['start_second']
        })
                
    return returned_contexts_answers

# interface

st.image("./decorations/logo.png", width=200)

st.title("Google and IBM Quantum Computing Videos Q&A Engine")
st.subheader("Explore knowledge from Google & IBM YouTube videos")

# Index and Retriever model setup
with st.spinner(text="Initializing index..."):
    index = init_pinecone()

with st.spinner(text="Initializing Retriever model..."):
    retriever = init_retriever()

#question_answerer = load_qa_pipeline()

query = st.text_input("Question:", help="enter your question here")
filter_origin = st.multiselect(label="Content Origin", options=['Google', 'IBM'], help="Search from Google and/or IBM")
top_k = st.number_input("Nb of returned context:", 1, 3, help="Top 3 ranking contexts maximum")
search = st.button("Search")

if search and query != "":
    returned_contexts_answers = list(reconstruct_answered_context(index, query, top_k, filter_origin))

    if returned_contexts_answers and len(returned_contexts_answers)>0:
        merged_text = ''
        columns = st.columns(len(returned_contexts_answers))
        for i, col in enumerate(columns):
            with col:
                title = returned_contexts_answers[i]['title']
                text = returned_contexts_answers[i]['text']
                topics = returned_contexts_answers[i]['keywords']
                url = returned_contexts_answers[i]['url']
                origin = returned_contexts_answers[i]['origin']
                start = int(returned_contexts_answers[i]['start'])

                st.markdown(f'<div style="text-align: center;"><b>{title}</b></div>', unsafe_allow_html=True)
                displayed_text = '... ' + text + '...'
                st.caption(f'<div style="text-align: justify;">{displayed_text}</div>', unsafe_allow_html=True)
                st.caption(' ')
                st.video(url, start_time=start)
                st.markdown(f"**Topic**: {', '.join(topics)}")
                st.markdown(f"**Origin**: {origin}")

                merged_text += ' ' + text

        #answer = question_answerer(question=query, context=merged_text)['answer']
        #st.info(f'A possible answer to your question is: **{answer}**')

    else:
        st.warning("Nothing found, sorry...")


st.warning(
    """
    Disclaimer: the content shown in this page is the exclusive property of Google or IBM. 
    """
)

bottom_column_1, bottom_column_2, bottom_column_3 = st.columns([2, 6, 2])
with bottom_column_1:
    st.image("./decorations/logo-petroglyphs.jpg")

with bottom_column_2:
    st.caption(
        """
        If you are interested by adding similar Semantic Search Engine 
        to your content, please contact Petroglyphs NLP Consulting 
        (petroglyphs.nlp@gmail.com).
        """)
    st.caption(
        """
        You can also 
        ["Buy me a Coffee"](https://www.buymeacoffee.com/petroglyphx)
        to balance the cost associated to the Search indexing. Thanks.
        """
    )

with bottom_column_3:
    # buy me a coffee button
    st.image("./decorations/bmc_qr.png", width=100)
