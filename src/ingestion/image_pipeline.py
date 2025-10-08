"""
Image ingestion pipeline with query functionality.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
import base64
import io

logger = logging.getLogger(__name__)

# Image processing imports (with error handling)
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    logger.warning("PIL not available. Install with: pip install Pillow")

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available. Install with: pip install opencv-python")

# OCR imports (with error handling)
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. Install with: pip install pytesseract")

# Weaviate and embedding imports (with error handling)
try:
    import weaviate
    WEAVIATE_AVAILABLE = True
except ImportError:
    WEAVIATE_AVAILABLE = False
    logger.warning("Weaviate not available. Install with: pip install weaviate-client")

try:
    import open_clip
    import torch
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    logger.warning("OpenCLIP not available. Install with: pip install open-clip-torch")


class ImagePipeline:
    """Pipeline for ingesting and processing image documents."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the image pipeline with configuration."""
        self.config = config
        self.supported_formats = config.get('supported_formats', ['.jpg', '.jpeg', '.png'])
        self.max_file_size = config.get('max_file_size', 10 * 1024 * 1024)  # 10MB default
        self.max_image_size = config.get('max_image_size', (1024, 1024))  # Max width, height
        self.image_collection = None  # Will be set when connecting to vector database
        self.weaviate_client = None
        self.embedding_model = None
        self.ocr_reader = None
        self.collection_name = config.get('collection_name', 'image_documents')
        self.extract_text = config.get('extract_text', True)
        self.ocr_method = config.get('ocr_method', 'easyocr')  # 'easyocr' or 'tesseract'
    
    def process_file(self, file_path: str) -> Dict[str, Any]:
        """
        Process a single image file and extract content.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dictionary containing extracted image data and metadata
        """
        file_path = Path(file_path)
        
        # Check if file exists
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Check file extension
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self.max_file_size:
            raise ValueError(f"File too large: {file_size} bytes (max: {self.max_file_size})")
        
        # Load and process image
        image_data = self._load_image(file_path)
        
        # Extract text if enabled
        extracted_text = ""
        if self.extract_text:
            extracted_text = self._extract_text_from_image(image_data)
        
        # Generate image embedding
        image_embedding = self._generate_image_embedding(image_data)
        
        # Convert image to base64 for storage
        image_base64 = self._image_to_base64(image_data)
        
        return {
            'image_data': image_base64,
            'extracted_text': extracted_text,
            'image_embedding': image_embedding,
            'metadata': {
                'file_path': str(file_path),
                'file_size': file_size,
                'file_type': file_path.suffix,
                'image_size': image_data.size,
                'image_mode': image_data.mode,
                'text_length': len(extracted_text)
            }
        }
    
    def process_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        Process all supported image files in a directory.
        
        Args:
            directory_path: Path to the directory containing image files
            
        Returns:
            List of processed image documents
        """
        directory = Path(directory_path)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        processed_files = []
        
        for file_path in directory.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in self.supported_formats:
                try:
                    result = self.process_file(str(file_path))
                    processed_files.append(result)
                except Exception as e:
                    logger.warning(f"Skipping file {file_path} due to error: {str(e)}")
        
        return processed_files
    
    def image_uris(self, query_text: str, max_distance: Optional[float] = None, max_results: int = 5) -> List[str]:
        """
        Query image collection and return filtered results.
        
        Args:
            query_text: Text to search for
            max_distance: Maximum distance threshold for filtering results
            max_results: Maximum number of results to return
            
        Returns:
            List of filtered image URIs
        """
        if self.image_collection is None:
            raise ValueError("Image collection not initialized. Please connect to vector database first.")
        
        try:
            # Query the image collection
            results = self.image_collection.query(
                query_texts=[query_text],
                n_results=max_results,
                include=['uris', 'distances']
            )
            
            # Filter results based on distance threshold
            filtered_uris = []
            for uri, distance in zip(results['uris'][0], results['distances'][0]):
                if max_distance is None or distance <= max_distance:
                    filtered_uris.append(uri)
            
            logger.info(f"Query '{query_text}' returned {len(filtered_uris)} results")
            return filtered_uris
            
        except Exception as e:
            logger.error(f"Error querying image collection: {str(e)}")
            raise
    
    def set_image_collection(self, collection):
        """
        Set the image collection for querying.
        
        Args:
            collection: Vector database collection object
        """
        self.image_collection = collection
        logger.info("Image collection initialized")
    
    def _load_image(self, file_path: Path) -> Image.Image:
        """
        Load and preprocess an image file.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        if not PIL_AVAILABLE:
            raise ImportError("PIL not available. Install with: pip install Pillow")
        
        try:
            # Load image
            image = Image.open(file_path)
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize if too large
            if image.size[0] > self.max_image_size[0] or image.size[1] > self.max_image_size[1]:
                image.thumbnail(self.max_image_size, Image.Resampling.LANCZOS)
            
            return image
            
        except Exception as e:
            logger.error(f"Error loading image {file_path}: {str(e)}")
            raise
    
    def _extract_text_from_image(self, image: Image.Image) -> str:
        """
        Extract text from image using OCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        if not self.extract_text:
            return ""
        
        try:
            if self.ocr_method == 'easyocr' and EASYOCR_AVAILABLE:
                return self._extract_text_easyocr(image)
            elif self.ocr_method == 'tesseract' and TESSERACT_AVAILABLE:
                return self._extract_text_tesseract(image)
            else:
                logger.warning(f"OCR method {self.ocr_method} not available")
                return ""
                
        except Exception as e:
            logger.error(f"Error extracting text from image: {str(e)}")
            return ""
    
    def _extract_text_easyocr(self, image: Image.Image) -> str:
        """
        Extract text using EasyOCR.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        if not EASYOCR_AVAILABLE:
            raise ImportError("EasyOCR not available. Install with: pip install easyocr")
        
        if self.ocr_reader is None:
            self.ocr_reader = easyocr.Reader(['en'])
        
        # Convert PIL image to numpy array
        import numpy as np
        image_array = np.array(image)
        
        # Extract text
        results = self.ocr_reader.readtext(image_array)
        
        # Combine all text
        text_parts = [result[1] for result in results]
        return ' '.join(text_parts)
    
    def _extract_text_tesseract(self, image: Image.Image) -> str:
        """
        Extract text using Tesseract.
        
        Args:
            image: PIL Image object
            
        Returns:
            Extracted text string
        """
        if not TESSERACT_AVAILABLE:
            raise ImportError("Tesseract not available. Install with: pip install pytesseract")
        
        try:
            text = pytesseract.image_to_string(image)
            return text.strip()
        except Exception as e:
            logger.error(f"Tesseract error: {str(e)}")
            return ""
    
    def _generate_image_embedding(self, image: Image.Image) -> List[float]:
        """
        Generate embedding for an image using OpenCLIP.
        
        Args:
            image: PIL Image object
            
        Returns:
            List of embedding values
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("OpenCLIP not available. Install with: pip install open-clip-torch")
        
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized. Call initialize_embedding_model() first.")
        
        try:
            # Preprocess image for the model
            model, _, preprocess = self.embedding_model
            
            # Preprocess the image
            image_tensor = preprocess(image).unsqueeze(0)
            
            # Generate embedding
            with torch.no_grad():
                image_features = model.encode_image(image_tensor)
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embedding = image_features.squeeze(0).tolist()
            
            return embedding
            
        except Exception as e:
            logger.error(f"Failed to generate image embedding: {str(e)}")
            raise
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """
        Convert PIL image to base64 string.
        
        Args:
            image: PIL Image object
            
        Returns:
            Base64 encoded image string
        """
        try:
            # Convert to bytes
            buffer = io.BytesIO()
            image.save(buffer, format='JPEG', quality=85)
            image_bytes = buffer.getvalue()
            
            # Encode to base64
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            return image_base64
            
        except Exception as e:
            logger.error(f"Error converting image to base64: {str(e)}")
            raise
    
    def connect_to_weaviate(self) -> bool:
        """
        Connect to Weaviate and initialize the image collection.
        
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
            self.image_collection = self.weaviate_client.collections.get(self.collection_name)
            
            logger.info(f"Connected to Weaviate at {host}:{port}")
            logger.info(f"Using collection: {self.collection_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to Weaviate: {str(e)}")
            return False
    
    def initialize_embedding_model(self, model_name: str = 'ViT-B-32') -> bool:
        """
        Initialize the image embedding model.
        
        Args:
            model_name: Name of the OpenCLIP model
            
        Returns:
            True if initialization successful, False otherwise
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("OpenCLIP not available. Install with: pip install open-clip-torch")
        
        try:
            # Load the model
            model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained='openai')
            self.embedding_model = (model, _, preprocess)
            
            logger.info(f"Initialized image embedding model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {str(e)}")
            return False
    
    def store_image_in_weaviate(self, image_data: Dict[str, Any]) -> bool:
        """
        Store a processed image in Weaviate.
        
        Args:
            image_data: Dictionary containing image data and metadata
            
        Returns:
            True if storage successful, False otherwise
        """
        if self.image_collection is None:
            raise ValueError("Weaviate collection not initialized. Call connect_to_weaviate() first.")
        
        if self.embedding_model is None:
            raise ValueError("Embedding model not initialized. Call initialize_embedding_model() first.")
        
        try:
            # Prepare image document for Weaviate
            weaviate_document = {
                'image_data': image_data['image_data'],
                'extracted_text': image_data['extracted_text'],
                'file_path': image_data['metadata']['file_path'],
                'file_size': image_data['metadata']['file_size'],
                'file_type': image_data['metadata']['file_type'],
                'image_size': str(image_data['metadata']['image_size']),
                'image_mode': image_data['metadata']['image_mode'],
                'text_length': image_data['metadata']['text_length']
            }
            
            # Store in Weaviate
            result = self.image_collection.data.insert(
                properties=weaviate_document,
                vector=image_data['image_embedding']
            )
            
            logger.info(f"Stored image in Weaviate: {result}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store image in Weaviate: {str(e)}")
            return False
    
    def process_and_store_file(self, file_path: str) -> bool:
        """
        Process an image file and store it in Weaviate.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Process the file
            image_data = self.process_file(file_path)
            
            # Store in Weaviate
            return self.store_image_in_weaviate(image_data)
            
        except Exception as e:
            logger.error(f"Failed to process and store image file {file_path}: {str(e)}")
            return False
    
    def close_weaviate_connection(self):
        """Close the Weaviate connection."""
        if self.weaviate_client:
            self.weaviate_client.close()
            logger.info("Closed Weaviate connection")
