"""nltk word tokens without punctuation, case kept (for cased vocabularies like word2vec-google-news-300).

raw text:  He was reunited with his parents, who also lived at the zoo.
tokenized: ['He', 'was', 'reunited', 'with', 'his', 'parents', ',', 'who', 'also', 'lived', 'at', 'the', 'zoo', '.']
filtered:  ['He', 'was', 'reunited', 'with', 'his', 'parents', 'who', 'also', 'lived', 'at', 'the', 'zoo']
"""
import nltk

# ~/nltk_data and are fetched at runtime. word_tokenize first splits the text into sentences
# with the Punkt model ("punkt_tab"), so without this line it raises a LookupError. 

# punctuation not needed, case sensitivity as in model as discussed in the project discussions

nltk.download("punkt_tab", quiet=True)

CONFIG = {
    "tokenizer": "nltk.word_tokenize",  # label: the function used in preprocess() below
    "lowercase": False,  # google news vectors are case sensitive -> keep the surface form
    "remove_punctuation": True, # as discussed in project discussions
    "remove_stopwords": False, # as discussed in project discussion
    "normalization": "none",  # no stemming, no lemmatization according to word2vec
    "truncation": "none", # won't make a difference at that size i am pretty sure
}


def preprocess(text):
    # Turn a raw sentence into a list of tokens for word2vec. operation is elementary -> on the most atomic element to tokenize in this model its on word level
    tokens = nltk.word_tokenize(text)
    # Since we use punkt_tab tokenization - punctuation might also occur in the tokenized results, therefore we remove them by filtering via isalnum
    return [token for token in tokens if any(character.isalnum() for character in token)]
