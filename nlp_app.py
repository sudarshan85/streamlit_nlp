#!/usr/bin/env python

import streamlit as st

from ttictoc import TicToc
import spacy
from textblob import TextBlob
from gensim.summarization import summarize as gensim_summarizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

@st.cache(allow_output_mutation=True)
def load_spacy(model='en_core_web_sm'):
  nlp = spacy.load(model)
  return nlp

def text_analyzer(sp_model, text):
  doc = sp_model(text)
  return [(f"Token: {token.text}, Lemma: {token.lemma_}") for token in doc]

def entity_extractor(sp_model, text):
  doc = sp_model(text)
  return {ent.text: ent.label_ for ent in doc.ents}

def sumy_summarizer(text):
  parser = PlaintextParser.from_string(text, Tokenizer('english'))
  lex_summarizer = LexRankSummarizer()
  return ' '.join([str(sent) for sent in lex_summarizer(parser.document, 3)])

def main():
  st.title("NLP App with Streamlit")
  nlp = load_spacy()

  st.subheader("Basic NLP Tasks")
  text = st.text_area("Enter Text and click on application", "Type Here")
  
  if st.button("Show Tokens & Lemmas"):
    st.json(text_analyzer(nlp, text))

  if st.button("Extract Entities"):
    st.json(entity_extractor(nlp, text))

  if st.button("Show sentiment"):
    blob = TextBlob(text)
    st.write(blob.sentiment)
    st.info("Polarity is between -1 (negative) and 1 (positive) indicating the type of sentiment\nSubjectivity is between 0 (objective) and 1 (subjective) indicating the bias of the sentiment")

  st.subheader("Text Summarizer")
  text = st.text_area("Enter Text", "Type Here")
  
  summerizer_type = st.selectbox("Summarizer", ("gensim", "sumy"))
  if summerizer_type == 'gensim':
    summarizer = gensim_summarizer
    st.info("Using gensim summarizer...")
  elif summerizer_type == 'sumy':
    summarizer = sumy_summarizer
    st.info("Using sumy summarizer...")

  if st.button("Summarize"):
    st.success(summarizer(text))

  st.sidebar.subheader("App Details")
  st.sidebar.text("Demo of using Streamlit to \ncreate basic NLP tasks")

  if st.button("Thanks!"):
    st.balloons()
  
if __name__ == "__main__":
  main()