# Basic NLP App using Streamlit

This is a project which uses [Streamlit](https://docs.streamlit.io/) to deploy an NLP app that does basic NLP tasks. It was created based on a series of tutorial videos on Youtube by [Jesse E. Agbe](https://github.com/Jcharis/). All of his Streamlit tutorials can be found [here](https://www.youtube.com/playlist?list=PLJ39kWiJXSixyRMcn3lrbv8xI8ZZoYNZU). 

The tasks that are implemented are:
1. Tokenization and Lemmatization (using [spaCy](https://spacy.io/usage))
2. Named Entity Recognition (using spaCy)
3. Sentiment Analysis (using [TextBlob](https://textblob.readthedocs.io/en/dev/))
4. Text summarization (using [gensim](https://radimrehurek.com/gensim/auto_examples/index.html) and  [sumy](https://github.com/miso-belica/sumy))

Here is a list of the tutorial videos that was used to create this app:
1. [Building a NLP App with Streamlit,Spacy and Python (NER,Sentiment Analyser,Summarizer)](https://youtu.be/6acv9LL6gHg)
2. [How To Deploy Streamlit Apps (Using Heroku)](https://youtu.be/skpiLtEN3yk)
3. [Building a Summarizer and Named Entity Checker NLP App with Streamlit and SpaCy](https://youtu.be/ftySGn13k8w)

The app is hosted on [Heroku](https://www.heroku.com/home) and you can access it [here](https://streamlit-nlp.herokuapp.com/). It is hosted on a free account and server will sleep after 30 minutes of inactivity. So please give it 10-15 seconds to wake up.