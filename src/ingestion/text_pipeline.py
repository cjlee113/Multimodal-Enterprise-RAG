"""
Text ingestion pipeline with query functionality.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# Weaviate and embedding imports (with error handling)
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("Weaviate not available. Install with: pip install weaviate-client")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("SentenceTransformers not available. Install with: pip install sentence-transformers")


class TextPipeline:
    """Pipeline for ingesting and processing text documents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the text pipeline with configuration."""
        self.config = config
        self.supported_formats = config.get('supported_formats', ['.txt', '.md', '.pdf'])
        self.max_file_size = config.get('max_file_size', 1024 * 1024)  # 1MB default
        self.encoding = config.get('encoding', 'utf-8')
        self.text_collection = None  # Will be set when connecting to vector database
        self.weaviate_client = None
        self.embedding_model = None
        self.collection_name = config.get('collection_name', 'text_documents')
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single text file and extract content.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        if file_path.suffix not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Read file content based on file type
        if file_path.suffix == '.pdf':
            content = self._extract_pdf_content(file_path)
        else:
            try:
                with open(file_path, 'r', encoding=self.encoding) as f:
                    content = f.read()
            except UnicodeDecodeError:
                # Try with different encoding if UTF-8 fails
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
        
        # Basic preprocessing
        cleaned_content = self._clean_text(content)
        
        return {
            'content': cleaned_content,
            'metadata': {
                'file_path': str(file_path),
                'file_size': file_size,
                'file_type': file_path.suffix,
                'original_length': len(content),
                'cleaned_length': len(cleaned_content)
            }
        }
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported text files in a directory.
        
        Args:
            directory_path: Path to the directory containing text files
            
        Returns:
            List of processed documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        processed_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix in self.supported_formats:
                try:
                    result = self.process_file(str(file_path))
                    processed_files.append(result)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
        
        return processed_files
    
    def text_uris(self, query_text: str, max_distance: Optional[float] = None, max_results: int = 5) -> List[str]:
        """
        Query text collection and return filtered results.
        
        Args:
            query_text: Text to search for
            max_distance: Maximum distance threshold for filtering results
            max_results: Maximum number of results to return
            
        Returns:
            List of filtered text documents
        """
        if self.text_collection is None:
            raise ValueError("Text collection not initialized. Please connect to vector database first.")
        
        try:
            # Query the text collection
            results = self.text_collection.query(
                query_texts=[query_text],
                n_results=max_results,
                include=['documents', 'distances']
            )
            
            # Filter results based on distance threshold
            filtered_texts = []
            for doc, distance in zip(results['documents'][0], results['distances'][0]):
                if max_distance is None or distance <= max_distance:
                    filtered_texts.append(doc)
            
            logger.info(f"Query '{query_text}' returned {len(filtered_texts)} results")
            return filtered_texts
            
        except Exception as e:
            logger.error(f"Error querying text collection: {str(e)}")
            raise
    
    def set_text_collection(self, collection):
        """
        Set the text collection for querying.
        
        Args:
            collection: Vector database collection object
        """
        self.text_collection = collection
        logger.info("Text collection initialized")
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text content
        """
        # Remove extra whitespace and normalize newlines
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _tokenize_text(self, text: str) -> List[str]:
        """
        Basic tokenization of text.
        
        Args:
            text: Text to tokenize
            
        Returns:
            List of tokens
        """
        # Simple whitespace tokenization
        tokens = text.split()
        
        # Remove empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def _chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Maximum size of each chunk
            overlap: Number of characters to overlap between chunks
            
        Returns:
            List of text chunks
        """
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(text):
                break
        
        return chunks
    
    def _extract_pdf_content(self, file_path: Path) -> str:
        """
        Extract text content from PDF files.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Extracted text content
        """
        try:
            import PyPDF2
        except ImportError:
            raise ImportError("PyPDF2 not available. Install with: pip install PyPDF2")
        
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_content = ""
                
                for page_num in range(len(pdf_reader.pages)):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
                
                return text_content
                
        except Exception as e:
            logger.error(f"Error extracting PDF content from {file_path}: {str(e)}")
            raise
    
    def connect_to_weaviate(self) -> bool:
        """
        Connect to Weaviate and initialize the text collection.
        
        Returns:
            True if connection successful, False otherwise
        """
        if not WEAVIATE_AVAILABLE:
            raise ImportError("Weaviate not available. Install with: pip install weaviate-client")
        
        try:
            # Get Weaviate configuration
            weaviate_config = self.config.get('weaviate_config', {})
            host = weaviate_config.get('host', 'localhost')
            port = weaviate_config.get('port', 8080)
            
            # Connect to Weaviate
            self.weaviate_client = weaviate.connect_to_local(
                host=host,
                port=port
            )
            
            # Get or create collection
            self.text_collection = self.weaviate_client.collections.get(self.collection_name)
            
            logger.info(f"Connected to Weaviate at {host}:{port}")
            logger.info(f"Using collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            return False
    
    def initialize_embedding_model(self, model_name: str = 'all-MiniLM-L6-v2') -> bool:
        """
        Initialize the embedding model.
        
        Args:
            model_name: Name of the sentence transformer model
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("SentenceTransformers not available. Install with: pip install sentence-transformers")
        
        try:
            self.embedding_model = SentenceTransformer(model_name)
            logger.info(f"Initialized embedding model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            return False
    
    def _generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a text.
        
        Args:
            text: Text to generate embedding for
            
        Returns:
            List of embedding values
        """
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized. Call initialize_embedding_model() first.")
        
        try:
            embedding = self.embedding_model.encode([text])[0]
            return embedding.tolist()
            
        except Exception as e:
            logger.error(f"Failed to generate embedding: {str(e)}")
            raise
    
    def store_document_in_weaviate(self, document_data: Dict[str, Any]) -> bool:
        """
        Store a processed document in Weaviate.
        
        Args:
            document_data: Dictionary containing document content and metadata
            
        Returns:
            True if storage successful, False otherwise
        """
        if self.text_collection is None:
            raise ValueError("Weaviate collection not initialized. Call connect_to_weaviate() first.")
        
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized. Call initialize_embedding_model() first.")
        
        try:
            # Generate embedding
            content = document_data['content']
            embedding = self._generate_embedding(content)
            
            # Prepare document for Weaviate
            weaviate_document = {
                'content': content,
                'file_path': document_data['metadata']['file_path'],
                'file_size': document_data['metadata']['file_size'],
                'file_type': document_data['metadata']['file_type'],
                'original_length': document_data['metadata']['original_length'],
                'cleaned_length': document_data['metadata']['cleaned_length']
            }
            
            # Store in Weaviate
            result = self.text_collection.data.insert(
                properties=weaviate_document,
                vector=embedding
            )
            
            logger.info(f"Stored document in Weaviate: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store document in Weaviate: {str(e)}")
            return False
    
    def process_and_store_file(self, file_path: str) -> bool:
        """
        Process a file and store it in Weaviate.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process the file
            document_data = self.process_file(file_path)
            
            # Store in Weaviate
            return self.store_document_in_weaviate(document_data)
            
        except Exception as e:
            logger.error(f"Failed to process and store file {file_path}: {str(e)}")
            return False
    
    def close_weaviate_connection(self):
        """Close the Weaviate connection."""
        if self.weaviate_client:
            self.weaviate_client.close()
            logger.info("Closed Weaviate connection")
