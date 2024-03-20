import re
import nltk
import spacy

def chunk_text(text, chunk_size=256, chunk_overlap=20, method="sentence", **kwargs):
    """
    Chunk the input text into smaller segments.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The target size of each chunk in tokens.
        chunk_overlap (int): The number of overlapping tokens between chunks.
        method (str): The chunking method to use. Options: "fixed", "sentence", "nltk", "spacy", "recursive", "markdown", "latex".
        **kwargs: Additional keyword arguments specific to the chosen chunking method.

    Returns:
        list: A list of chunked text segments.
    """
    if method == "fixed":
        return fixed_size_chunking(text, chunk_size, chunk_overlap)
    elif method == "sentence":
        return sentence_chunking(text)
    elif method == "nltk":
        return nltk_chunking(text)
    elif method == "spacy":
        return spacy_chunking(text)
    else:
        raise ValueError(f"Unsupported chunking method: {method}")

def fixed_size_chunking(text, chunk_size, chunk_overlap):
    """
    Perform fixed-size chunking on the input text.

    Args:
        text (str): The input text to be chunked.
        chunk_size (int): The target size of each chunk in tokens.
        chunk_overlap (int): The number of overlapping tokens between chunks.

    Returns:
        list: A list of chunked text segments.
    """
    tokens = text.split()
    chunks = []
    for i in range(0, len(tokens), chunk_size - chunk_overlap):
        chunk = tokens[i:i + chunk_size]
        chunks.append(" ".join(chunk))
    return chunks

def sentence_chunking(text):
    """
    Perform sentence-level chunking on the input text.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of chunked text segments.
    """
    sentences = re.split(r'(?<=[^A-Z].[.?]) +(?=[A-Z])', text)
    return sentences

def nltk_chunking(text):
    """
    Perform chunking using NLTK's sentence tokenizer.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of chunked text segments.
    """
    nltk.download('punkt', quiet=True)
    sentences = nltk.sent_tokenize(text)
    return sentences

def spacy_chunking(text):
    """
    Perform chunking using spaCy's sentence segmentation.

    Args:
        text (str): The input text to be chunked.

    Returns:
        list: A list of chunked text segments.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    sentences = [sent.text for sent in doc.sents]
    return sentences