#!/usr/bin/env python

import requests
import spacy

import numpy as np
import streamlit as st

from bs4 import BeautifulSoup
from ttictoc import TicToc

from spacy import displacy
from textblob import TextBlob
from gensim.summarization import summarize as gensim_summarizer

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

HTML_WRAPPER = """<div style="overflow-x: auto; border: 1px solid #e6e9ef; border-radius: 0.25rem; padding: 1rem">{}</div>"""

@st.cache(allow_output_mutation=True)
def load_spacy(model='en_core_web_sm'):
  nlp = spacy.load(model)
  return nlp

@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def analyze_text(sp_model, text):
  return sp_model(text)

def fetch_text(text):
  try:
    page = requests.get(text)
    soup = BeautifulSoup(page.text)
    return ' '.join(map(lambda p: p.text, soup.find_all('p')))
  except (requests.exceptions.InvalidURL, requests.exceptions.MissingSchema):
    return text

def sumy_summarizer(text):
  parser = PlaintextParser.from_string(text, Tokenizer('english'))
  lex_summarizer = LexRankSummarizer()
  return ' '.join([str(sent) for sent in lex_summarizer(parser.document, 3)])

def main():
  st.title("NLP App with Streamlit")
  nlp = load_spacy()

  text = fetch_text(st.text_area("Enter Text (or URL) and select application from sidebar", "Type Here"))
  apps = ['Show tokens & lemmas', 'Extract Entities', 'Show sentiment', 'Summarize text']

  choice = st.sidebar.selectbox("Select Application", apps)
  if choice == "Show tokens & lemmas":
    if st.button("Tokenize"):
      st.info("Using spaCy for tokenization and lemmatization")
      st.json([(f"Token: {token.text}, Lemma: {token.lemma_}") for token in analyze_text(nlp, text)])
  elif choice == 'Extract Entities':
    if st.button("Extract"):
      st.info("Using spaCy for NER")      
      doc = analyze_text(nlp, text)
      html = displacy.render(doc, style='ent')
      html = html.replace('\n\n', '\n')
      st.write(html, unsafe_allow_html=True)
  elif choice == "Show sentiment":
    if st.button("Analyze"):
      st.info("Using TextBlob for sentiment analysis")
      blob = TextBlob(text)
      sentiment = {
        'polarity': np.round(blob.sentiment[0], 3),
        'subjectivity': np.round(blob.sentiment[1], 3),
      }
      st.write(sentiment)
      st.info("Polarity is between -1 (negative) and 1 (positive) indicating the type of sentiment\nSubjectivity is between 0 (objective) and 1 (subjective) indicating the bias of the sentiment")
  elif choice == "Summarize text":    
    summarizer_type = st.sidebar.selectbox("Select Summarizer", ['Gensim', 'Sumy Lex Rank'])
    if summarizer_type == 'Gensim':
      summarizer = gensim_summarizer
    elif summarizer_type == 'Sumy Lex Rank':
      summarizer = sumy_summarizer

    if st.button(f"Summarize using {summarizer_type}"):
      st.success(summarizer(text))
  
if __name__ == "__main__":
  main()