"""
Generate vector embeddings for clinical trial text.

I'm using the all-MiniLM-L6-v2 model from SentenceTransformers.
It's small (80MB), fast, and good enough for my search use case.
Bigger models exist but they're slower and I don't need them here.

How this works:
- The model turns any text into a 384-dimensional vector
- Similar texts get similar vectors
- "heart attack" and "myocardial infarction" will be close together
- I store these vectors in OpenSearch alongside the text data
"""

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# This model name is specific — don't change it unless you also
# change the vector dimension (384) in the OpenSearch mapping
MODEL_NAME = "all-MiniLM-L6-v2"
VECTOR_DIMENSION = 384

# Load the model once and reuse it (loading takes a few seconds)
_model = None


def get_model():
    """
    Load the embedding model. Only loads once — after that, returns
    the cached version. This pattern is called "lazy loading".
    """
    global _model
    if _model is None:
        print("  Loading embedding model (first time only)...")
        _model = SentenceTransformer(MODEL_NAME)
        print("  Model loaded.")
    return _model


def generate_embedding(text):
    """
    Turn a single piece of text into a vector.

    Input: "type 2 diabetes with kidney complications"
    Output: [0.023, -0.114, 0.871, ...] (384 numbers)
    """
    model = get_model()
    # encode() returns a numpy array, tolist() converts to plain Python list
    # OpenSearch needs a plain list, not a numpy array
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


def generate_embeddings_batch(texts, batch_size=64):
    """
    Turn a list of texts into vectors, processing in batches.

    Batching is faster than one-at-a-time because the GPU/CPU
    can process multiple texts in parallel.

    Input: ["text 1", "text 2", "text 3", ...]
    Output: [[0.1, 0.2, ...], [0.3, 0.4, ...], [0.5, 0.6, ...], ...]
    """
    model = get_model()
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="  Generating embeddings"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch, show_progress_bar=False)
        all_embeddings.extend(batch_embeddings.tolist())

    return all_embeddings