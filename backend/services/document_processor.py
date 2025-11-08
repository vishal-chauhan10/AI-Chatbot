#!/usr/bin/env python3
"""
Document Processing Pipeline for Multilingual Spiritual Content
==============================================================

This service processes raw text documents (especially from Telegram exports) and
prepares them for optimal RAG (Retrieval-Augmented Generation) performance.

Key Features:
1. Text Cleaning and Normalization
2. Intelligent Chunking for Better Retrieval
3. Language Detection and Processing
4. Content Quality Enhancement
5. Metadata Enrichment
6. Speaker and Topic Extraction

Why This Phase is Critical:
- Raw Telegram exports have poor formatting
- Long documents need chunking for better retrieval
- Multilingual content needs special handling
- Quality content leads to better RAG responses
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import unicodedata

# Language detection
try:
    from langdetect import detect, detect_langs
    LANGDETECT_AVAILABLE = True
except ImportError:
    LANGDETECT_AVAILABLE = False
    logging.warning("langdetect not available, using fallback language detection")

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """
    Comprehensive document processing pipeline for multilingual spiritual content.
    
    Processes raw text through multiple stages:
    1. Text cleaning and normalization
    2. Language detection and segmentation
    3. Intelligent chunking
    4. Content enhancement
    5. Metadata extraction and enrichment
    """
    
    def __init__(self):
        """Initialize the document processor with configuration."""
        
        # Chunking configuration
        self.max_chunk_size = 800  # Maximum characters per chunk
        self.min_chunk_size = 100  # Minimum characters per chunk
        self.overlap_size = 100    # Overlap between chunks
        
        # Language patterns
        self.gujarati_pattern = re.compile(r'[\u0A80-\u0AFF]+')
        self.hindi_pattern = re.compile(r'[\u0900-\u097F]+')
        self.english_pattern = re.compile(r'[a-zA-Z]+')
        
        # Content patterns
        self.speaker_patterns = [
            re.compile(r'^([A-Za-z\s]+):\s*(.+)', re.MULTILINE),
            re.compile(r'^([ગુજરાતી\s]+):\s*(.+)', re.MULTILINE),
            re.compile(r'વક્તા\s*[-:]?\s*([^\n]+)', re.IGNORECASE),
            re.compile(r'Host\s*[-:]?\s*([^\n]+)', re.IGNORECASE),
        ]
        
        # Topic extraction patterns
        self.topic_patterns = [
            re.compile(r'Topic\s*[-:]?\s*([^\n]+)', re.IGNORECASE),
            re.compile(r'વિષય\s*[-:]?\s*([^\n]+)', re.IGNORECASE),
        ]
        
        # Quality filters
        self.noise_patterns = [
            re.compile(r'\[.*?\]'),  # Remove bracketed content
            re.compile(r'\(.*?\)'),  # Remove parenthetical content (optional)
            re.compile(r'https?://\S+'),  # Remove URLs
            re.compile(r'@\w+'),  # Remove mentions
            re.compile(r'#\w+'),  # Remove hashtags
        ]
        
        logger.info("📝 Document Processor initialized")
    
    def process_document(self, content: str, document_id: str, metadata: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Process a raw document through the complete pipeline.
        
        Args:
            content: Raw document content
            document_id: Unique document identifier
            metadata: Existing metadata (will be enhanced)
            
        Returns:
            List of processed document chunks with enhanced metadata
        """
        logger.info(f"📄 Processing document: {document_id}")
        
        if not content or len(content.strip()) < self.min_chunk_size:
            logger.warning(f"⚠️ Document {document_id} too short or empty")
            return []
        
        try:
            # Stage 1: Clean and normalize text
            cleaned_content = self._clean_text(content)
            
            # Stage 2: Detect and analyze language
            language_info = self._analyze_language(cleaned_content)
            
            # Stage 3: Extract speakers and structure
            speakers_info = self._extract_speakers(cleaned_content)
            
            # Stage 4: Extract topics and themes
            topics_info = self._extract_topics(cleaned_content)
            
            # Stage 5: Intelligent chunking
            chunks = self._chunk_content(cleaned_content, document_id)
            
            # Stage 6: Enhance each chunk with metadata
            processed_chunks = []
            for i, chunk in enumerate(chunks):
                enhanced_chunk = self._enhance_chunk(
                    chunk, document_id, i, 
                    language_info, speakers_info, topics_info, metadata
                )
                processed_chunks.append(enhanced_chunk)
            
            logger.info(f"✅ Processed {document_id}: {len(processed_chunks)} chunks created")
            return processed_chunks
            
        except Exception as e:
            logger.error(f"❌ Error processing document {document_id}: {e}")
            return []
    
    def _clean_text(self, text: str) -> str:
        """
        Clean and normalize raw text content.
        
        Steps:
        1. Unicode normalization
        2. Remove noise patterns
        3. Normalize whitespace
        4. Fix common formatting issues
        """
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove noise patterns (selectively)
        # Keep some bracketed content that might be important
        text = re.sub(r'https?://\S+', '', text)  # Remove URLs
        text = re.sub(r'@\w+', '', text)  # Remove mentions
        
        # Fix common formatting issues
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        text = re.sub(r'^\s+|\s+$', '', text, flags=re.MULTILINE)  # Trim lines
        
        # Fix punctuation spacing
        text = re.sub(r'([.!?])\s*([A-Za-z])', r'\1 \2', text)
        
        return text.strip()
    
    def _analyze_language(self, text: str) -> Dict[str, Any]:
        """
        Analyze language composition of the text.
        
        Returns:
            Dictionary with language statistics and primary language
        """
        # Count characters by script
        gujarati_chars = len(self.gujarati_pattern.findall(text))
        hindi_chars = len(self.hindi_pattern.findall(text))
        english_chars = len(self.english_pattern.findall(text))
        total_chars = len(text)
        
        # Calculate percentages
        gujarati_pct = gujarati_chars / total_chars if total_chars > 0 else 0
        hindi_pct = hindi_chars / total_chars if total_chars > 0 else 0
        english_pct = english_chars / total_chars if total_chars > 0 else 0
        
        # Determine primary language
        if gujarati_pct > 0.3:
            primary_language = "gujarati"
        elif hindi_pct > 0.3:
            primary_language = "hindi"
        elif english_pct > 0.5:
            primary_language = "english"
        else:
            primary_language = "mixed"
        
        # Use langdetect if available for additional confirmation
        detected_language = None
        if LANGDETECT_AVAILABLE and len(text) > 50:
            try:
                detected_language = detect(text)
            except:
                pass
        
        return {
            "primary_language": primary_language,
            "detected_language": detected_language,
            "gujarati_percentage": gujarati_pct,
            "hindi_percentage": hindi_pct,
            "english_percentage": english_pct,
            "is_multilingual": gujarati_pct > 0.1 and english_pct > 0.1
        }
    
    def _extract_speakers(self, text: str) -> Dict[str, Any]:
        """
        Extract speaker information from the text.
        
        Returns:
            Dictionary with speaker names and their content
        """
        speakers = {}
        speaker_count = {}
        
        for pattern in self.speaker_patterns:
            matches = pattern.findall(text)
            for match in matches:
                if isinstance(match, tuple):
                    speaker_name = match[0].strip()
                    content = match[1].strip()
                else:
                    speaker_name = match.strip()
                    content = ""
                
                if speaker_name and len(speaker_name) < 50:  # Reasonable speaker name length
                    if speaker_name not in speakers:
                        speakers[speaker_name] = []
                        speaker_count[speaker_name] = 0
                    
                    speakers[speaker_name].append(content)
                    speaker_count[speaker_name] += 1
        
        # Find primary speaker (most content)
        primary_speaker = max(speaker_count.keys(), key=lambda x: speaker_count[x]) if speaker_count else None
        
        return {
            "speakers": list(speakers.keys()),
            "primary_speaker": primary_speaker,
            "speaker_count": len(speakers),
            "speaker_content": speakers
        }
    
    def _extract_topics(self, text: str) -> Dict[str, Any]:
        """
        Extract topics and themes from the text.
        
        Returns:
            Dictionary with extracted topics and themes
        """
        topics = []
        
        # Extract explicit topics
        for pattern in self.topic_patterns:
            matches = pattern.findall(text)
            for match in matches:
                topic = match.strip()
                if topic and len(topic) < 100:  # Reasonable topic length
                    topics.append(topic)
        
        # Extract themes from content (simple keyword extraction)
        themes = self._extract_themes(text)
        
        return {
            "explicit_topics": topics,
            "themes": themes,
            "primary_topic": topics[0] if topics else None
        }
    
    def _extract_themes(self, text: str) -> List[str]:
        """
        Extract themes using keyword analysis.
        
        Returns:
            List of identified themes
        """
        # Spiritual and philosophical keywords
        spiritual_keywords = {
            "meditation": ["ધ્યાન", "meditation", "मेडिटेशन"],
            "spirituality": ["આધ્યાત્મિકતા", "spirituality", "अध्यात्म"],
            "faith": ["વિશ્વાસ", "faith", "श्रद्धा"],
            "devotion": ["ભક્તિ", "devotion", "भक्ति"],
            "wisdom": ["જ્ઞાન", "wisdom", "ज्ञान"],
            "peace": ["શાંતિ", "peace", "शांति"],
            "happiness": ["આનંદ", "happiness", "आनंद"],
            "service": ["સેવા", "service", "सेवा"],
            "truth": ["સત્ય", "truth", "सत्य"],
            "love": ["પ્રેમ", "love", "प्रेम"]
        }
        
        themes = []
        text_lower = text.lower()
        
        for theme, keywords in spiritual_keywords.items():
            for keyword in keywords:
                if keyword.lower() in text_lower:
                    themes.append(theme)
                    break
        
        return themes
    
    def _chunk_content(self, content: str, document_id: str) -> List[Dict[str, Any]]:
        """
        Intelligently chunk content for optimal retrieval.
        
        Strategy:
        1. Try to break at natural boundaries (paragraphs, sentences)
        2. Maintain context with overlapping chunks
        3. Ensure minimum and maximum chunk sizes
        4. Preserve speaker transitions
        """
        if len(content) <= self.max_chunk_size:
            return [{
                "content": content,
                "chunk_index": 0,
                "chunk_type": "complete_document"
            }]
        
        chunks = []
        
        # First, try to split by paragraphs
        paragraphs = content.split('\n\n')
        
        current_chunk = ""
        chunk_index = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, finalize current chunk
            if len(current_chunk) + len(paragraph) > self.max_chunk_size and current_chunk:
                if len(current_chunk) >= self.min_chunk_size:
                    chunks.append({
                        "content": current_chunk.strip(),
                        "chunk_index": chunk_index,
                        "chunk_type": "paragraph_boundary"
                    })
                    chunk_index += 1
                
                # Start new chunk with overlap
                current_chunk = self._get_overlap(current_chunk) + paragraph
            else:
                # Add paragraph to current chunk
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add final chunk
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunks.append({
                "content": current_chunk.strip(),
                "chunk_index": chunk_index,
                "chunk_type": "final_chunk"
            })
        
        # If no valid chunks created, create one large chunk
        if not chunks:
            chunks.append({
                "content": content[:self.max_chunk_size],
                "chunk_index": 0,
                "chunk_type": "truncated"
            })
        
        logger.info(f"📄 Created {len(chunks)} chunks for {document_id}")
        return chunks
    
    def _get_overlap(self, text: str) -> str:
        """
        Get overlap text from the end of a chunk for context continuity.
        
        Args:
            text: Source text to get overlap from
            
        Returns:
            Overlap text for the next chunk
        """
        if len(text) <= self.overlap_size:
            return text
        
        # Try to find a good breaking point (sentence end)
        overlap_text = text[-self.overlap_size:]
        
        # Find the last sentence boundary
        sentence_end = max(
            overlap_text.rfind('.'),
            overlap_text.rfind('!'),
            overlap_text.rfind('?'),
            overlap_text.rfind('।')  # Devanagari sentence end
        )
        
        if sentence_end > self.overlap_size // 2:
            return overlap_text[sentence_end + 1:].strip()
        else:
            return overlap_text
    
    def _enhance_chunk(self, chunk: Dict[str, Any], document_id: str, chunk_index: int,
                      language_info: Dict, speakers_info: Dict, topics_info: Dict,
                      original_metadata: Dict = None) -> Dict[str, Any]:
        """
        Enhance a chunk with comprehensive metadata.
        
        Args:
            chunk: Raw chunk data
            document_id: Original document ID
            chunk_index: Index of this chunk
            language_info: Language analysis results
            speakers_info: Speaker extraction results
            topics_info: Topic extraction results
            original_metadata: Original document metadata
            
        Returns:
            Enhanced chunk with rich metadata
        """
        content = chunk["content"]
        
        # Create enhanced chunk ID
        chunk_id = f"{document_id}_chunk_{chunk_index:03d}"
        
        # Analyze this specific chunk
        chunk_language = self._analyze_language(content)
        chunk_speakers = self._extract_speakers(content)
        
        # Build comprehensive metadata (ChromaDB compatible - no lists)
        enhanced_metadata = {
            # Basic information
            "original_document_id": document_id,
            "chunk_index": chunk_index,
            "chunk_id": chunk_id,
            "chunk_type": chunk.get("chunk_type", "standard"),
            "content_length": len(content),
            
            # Language information
            "primary_language": chunk_language["primary_language"],
            "is_multilingual": str(chunk_language["is_multilingual"]),  # Convert bool to string
            "gujarati_percentage": round(chunk_language["gujarati_percentage"], 3),
            "english_percentage": round(chunk_language["english_percentage"], 3),
            
            # Content structure (convert lists to comma-separated strings)
            "speakers_in_chunk": ", ".join(chunk_speakers["speakers"]) if chunk_speakers["speakers"] else "Unknown",
            "primary_speaker": chunk_speakers["primary_speaker"] or "Unknown",
            "has_dialogue": str(len(chunk_speakers["speakers"]) > 1),  # Convert bool to string
            
            # Topics and themes (convert lists to comma-separated strings)
            "topics": ", ".join(topics_info["explicit_topics"]) if topics_info["explicit_topics"] else "General",
            "themes": ", ".join(topics_info["themes"]) if topics_info["themes"] else "General",
            "primary_topic": topics_info["primary_topic"] or "General",
            
            # Processing metadata
            "processed_at": datetime.utcnow().isoformat(),
            "processing_version": "1.0",
            "content_type": "processed_chunk"
        }
        
        # Merge with original metadata
        if original_metadata:
            enhanced_metadata.update(original_metadata)
        
        return {
            "chunk_id": chunk_id,
            "content": content,
            "metadata": enhanced_metadata,
            "quality_score": self._calculate_quality_score(content, enhanced_metadata)
        }
    
    def _calculate_quality_score(self, content: str, metadata: Dict) -> float:
        """
        Calculate a quality score for the chunk (0.0 to 1.0).
        
        Factors:
        - Content length (not too short, not too long)
        - Language clarity
        - Presence of speakers/structure
        - Topic relevance
        """
        score = 0.0
        
        # Length score (optimal around 300-600 characters)
        length = len(content)
        if 200 <= length <= 800:
            score += 0.3
        elif 100 <= length <= 1000:
            score += 0.2
        else:
            score += 0.1
        
        # Language clarity score
        if not metadata.get("is_multilingual", False):
            score += 0.2  # Single language is clearer
        else:
            score += 0.1  # Mixed language is still valuable
        
        # Structure score
        if metadata.get("primary_speaker"):
            score += 0.2  # Has identified speaker
        if metadata.get("topics"):
            score += 0.2  # Has identified topics
        
        # Content richness score
        if metadata.get("themes"):
            score += 0.1 * min(len(metadata["themes"]), 3)  # Up to 0.3 for themes
        
        return min(score, 1.0)
    
    def process_batch(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch.
        
        Args:
            documents: List of documents with 'content', 'document_id', and 'metadata'
            
        Returns:
            List of all processed chunks from all documents
        """
        all_chunks = []
        
        for doc in documents:
            chunks = self.process_document(
                content=doc.get("content", ""),
                document_id=doc.get("document_id", "unknown"),
                metadata=doc.get("metadata", {})
            )
            all_chunks.extend(chunks)
        
        logger.info(f"📦 Batch processed {len(documents)} documents → {len(all_chunks)} chunks")
        return all_chunks


# Global instance
document_processor = None

def get_document_processor() -> DocumentProcessor:
    """Get the global document processor instance."""
    global document_processor
    if document_processor is None:
        document_processor = DocumentProcessor()
    return document_processor

def initialize_document_processor() -> DocumentProcessor:
    """Initialize the document processor during application startup."""
    global document_processor
    document_processor = DocumentProcessor()
    logger.info("🚀 Document processor initialized")
    return document_processor
