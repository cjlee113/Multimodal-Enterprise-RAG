# Project: Multimodal Hybrid RAG (Weaviate, Neo4j, Streamlit)

I’m building a modular retrieval-augmented generation (RAG) system for text, images, audio, and video.
Tech stack: Python 3.10+, Weaviate, Neo4j, DeepEval, Streamlit, SentenceTransformers, EasyOCR/Tesseract.

Start by writing a unit test for the text ingestion/preprocessing pipeline (e.g., cleaning, simple tokenization).
Then, implement the minimal code to pass the test, using a file structure that separates ingestion, vector indexing, and retrieval.

After the text pipeline works, I’ll do the same for image, audio, and video.

Keep code modular, test-driven, and well-commented.

# Tech Stack

- **Python 3.10+**
- **Weaviate** (vector database)
- **Neo4j** (knowledge graph)
- **DeepEval** (retrieval evaluation/testing)
- **Streamlit** (UI and demo)
- **OCR**: Tesseract or EasyOCR (image/text extraction)
- **Transformers for Embeddings**: SentenceTransformers, OpenCLIP, etc.
- **Pytest or unittest** (Python testing framework)
- **Docker** (optional, for local Weaviate/Neo4j/test env)
- **JSON logging** (`logs/`)
- **Other libraries as needed**: requests, Pillow, langchain, etc.

> All technologies are pip-installable—see requirements.txt for details.
