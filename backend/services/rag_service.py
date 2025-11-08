"""
RAG (Retrieval-Augmented Generation) Service
===========================================

This service combines our vector database search with OpenAI's language models
to provide accurate, context-aware responses based on retrieved documents.

Key Concepts:
1. **Retrieval**: Search for relevant documents using vector similarity
2. **Augmentation**: Combine retrieved content with user query as context
3. **Generation**: Use OpenAI to generate response based on the context

Why RAG?
- Grounds AI responses in actual data (your Gujarati transcripts)
- Provides source attribution (which documents were used)
- Reduces hallucination by giving AI specific context to work with
- Enables multilingual conversations based on your content
"""

import logging
from typing import List, Dict, Any, Optional
import openai
from datetime import datetime
import json

from .vector_db import VectorDatabaseService
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class RAGService:
    """
    RAG Service that combines vector search with OpenAI generation.
    
    This service:
    - Retrieves relevant documents from our vector database
    - Assembles context from retrieved documents
    - Generates responses using OpenAI with the context
    - Provides source attribution and confidence scoring
    """
    
    def __init__(self, vector_db: VectorDatabaseService):
        """
        Initialize RAG service with vector database and OpenAI client.
        
        Args:
            vector_db: Vector database service for document retrieval
        """
        self.vector_db = vector_db
        self.openai_client = None
        self._initialize_openai()
        
        # RAG Configuration - Enhanced for better retrieval
        self.max_context_length = 6000  # Increased for richer context
        self.top_k_results = 8  # More documents for comprehensive coverage
        self.min_similarity_threshold = 0.05   # Lower threshold for broader recall
        
        logger.info("✅ RAG service initialized")
    
    def _initialize_openai(self):
        """
        Initialize OpenAI client with API key from environment.
        
        Why we use OpenAI:
        - High-quality multilingual text generation
        - Good at following instructions and context
        - Supports system prompts for behavior control
        - Reliable API with good error handling
        """
        try:
            if settings.openai_api_key:
                openai.api_key = settings.openai_api_key
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("✅ OpenAI client initialized")
            else:
                logger.warning("⚠️ OpenAI API key not found. RAG responses will be limited.")
                self.openai_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client: {e}")
            self.openai_client = None
    
    async def generate_response(
        self,
        query: str,
        user_language: Optional[str] = None,
        conversation_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate a RAG response for the user query.
        
        Process:
        1. Retrieve relevant documents from vector database
        2. Assemble context from retrieved documents
        3. Create system prompt with instructions
        4. Generate response using OpenAI
        5. Return response with sources and metadata
        
        Args:
            query: User's question or message
            user_language: Preferred response language (auto-detected if None)
            conversation_history: Previous messages for context
            
        Returns:
            Dict containing response, sources, confidence, and metadata
        """
        start_time = datetime.utcnow()
        
        try:
            # Step 1: Retrieve relevant documents
            logger.info(f"🔍 Retrieving documents for query: {query[:50]}...")
            retrieved_docs = await self._retrieve_documents(query)
            
            if not retrieved_docs:
                return self._create_no_context_response(query, user_language)
            
            # Step 2: Assemble context from retrieved documents
            context = self._assemble_context(retrieved_docs)
            
            # Step 3: Generate response using OpenAI
            if self.openai_client:
                response_text = await self._generate_openai_response(
                    query, context, user_language, conversation_history
                )
            else:
                response_text = self._generate_fallback_response(query, retrieved_docs)
            
            # Step 4: Calculate processing time and confidence
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            confidence = self._calculate_confidence(retrieved_docs, response_text)
            
            # Step 5: Format response with metadata
            return {
                "response": response_text,
                "language": user_language or "english",
                "confidence": confidence,
                "sources": [
                    {
                        "id": doc["id"],
                        "content": doc["content"][:200] + "..." if len(doc["content"]) > 200 else doc["content"],
                        "metadata": doc["metadata"],
                        "similarity_score": doc["similarity_score"]
                    }
                    for doc in retrieved_docs
                ],
                "processing_time": processing_time,
                "retrieved_docs_count": len(retrieved_docs),
                "context_length": len(context)
            }
            
        except Exception as e:
            logger.error(f"❌ RAG generation failed: {e}")
            return self._create_error_response(str(e))
    
    async def _retrieve_documents(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents from vector database.
        
        Why this approach:
        - Uses semantic similarity to find relevant content
        - Filters by minimum similarity threshold
        - Limits results to prevent context overflow
        - Returns documents with similarity scores for confidence calculation
        """
        try:
            # Search vector database for similar documents
            # First try the processed chunks collection (higher quality)
            results = self.vector_db.search_similar(
                query=query,
                collection_name="chunks",  # Use processed chunks for better quality
                n_results=self.top_k_results
            )
            
            # If no good results, also try original transcripts
            if len(results) < 2:
                backup_results = self.vector_db.search_similar(
                    query=query,
                    collection_name="transcripts",
                    n_results=self.top_k_results
                )
                results.extend(backup_results)
            
            # Filter by similarity threshold
            filtered_results = [
                doc for doc in results 
                if doc.get("similarity_score", 0) >= self.min_similarity_threshold
            ]
            
            logger.info(f"📊 Retrieved {len(filtered_results)} documents above threshold")
            return filtered_results
            
        except Exception as e:
            logger.error(f"❌ Document retrieval failed: {e}")
            return []
    
    def _assemble_context(self, retrieved_docs: List[Dict[str, Any]]) -> str:
        """
        Assemble rich context from retrieved documents for OpenAI.
        
        Enhanced strategy:
        - Rich metadata extraction from processed chunks
        - Clear session boundaries and attribution
        - Structured format for better AI understanding
        - Include themes and topics for context
        """
        if not retrieved_docs:
            return "No relevant documents found in the knowledge base."
        
        context_parts = []
        context_parts.append("=== RELEVANT SPIRITUAL SESSION TRANSCRIPTS ===\n")
        current_length = 0
        
        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get('metadata', {})
            similarity = doc.get('similarity_score', 0)
            
            # Extract enhanced metadata from processed chunks
            speaker = metadata.get('primary_speaker', metadata.get('speaker', 'Unknown'))
            topic = metadata.get('primary_topic', metadata.get('topic', 'General'))
            themes = metadata.get('themes', 'General')
            language = metadata.get('primary_language', metadata.get('language', 'mixed'))
            sabha_type = metadata.get('sabha_type', 'Akshar Sarjan Sabha')
            session_date = metadata.get('session_date', 'Unknown')
            
            # Format document with rich metadata for better AI understanding
            doc_context = f"""
📄 SESSION {i} (Relevance: {similarity:.1%})
🎯 Sabha Type: {sabha_type}
📅 Date: {session_date}  
🎤 Speaker: {speaker}
📝 Topic: {topic}
🏷️ Themes: {themes}
🌐 Language: {language}

💬 CONTENT:
{doc['content']}

{'='*50}
"""
            
            # Check context length limit
            if current_length + len(doc_context) > self.max_context_length:
                logger.info(f"📏 Context limit reached, using {i-1} documents")
                break
            
            context_parts.append(doc_context)
            current_length += len(doc_context)
        
        final_context = "\n".join(context_parts)
        
        # Debug logging to see what context is being assembled
        logger.info(f"📝 Assembled context length: {len(final_context)} characters")
        logger.info(f"📄 Context preview: {final_context[:500]}...")
        
        return final_context
    
    async def _generate_openai_response(
        self,
        query: str,
        context: str,
        user_language: Optional[str],
        conversation_history: Optional[List[Dict]]
    ) -> str:
        """
        Generate response using OpenAI with retrieved context.
        
        Why this prompt structure:
        - Clear instructions for the AI model
        - Emphasizes using only provided context
        - Handles multilingual responses appropriately
        - Includes conversation history for continuity
        """
        try:
            # Create system prompt with instructions
            system_prompt = self._create_system_prompt(user_language)
            
            # Create user message with context and query
            user_message = f"""
{context}

User Question: {query}

Based on the session transcripts provided above, please give a comprehensive answer. The context contains detailed information from Akshar Sarjan Sabha sessions with topics, speakers, and content.
"""
            
            # Prepare messages for OpenAI
            messages = [{"role": "system", "content": system_prompt}]
            
            # Add conversation history if provided
            if conversation_history:
                for msg in conversation_history[-6:]:  # Last 6 messages for context
                    messages.append({
                        "role": "user" if msg.get("isUser") else "assistant",
                        "content": msg.get("content", "")
                    })
            
            # Add current user message
            messages.append({"role": "user", "content": user_message})
            
            # Generate response using OpenAI
            logger.info("🤖 Generating response with OpenAI GPT-4o...")
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",  # Latest GPT-4 model with multimodal capabilities
                messages=messages,
                max_tokens=500,  # Reasonable response length
                temperature=0.7,  # Balanced creativity vs consistency
                top_p=0.9,
                frequency_penalty=0.1,
                presence_penalty=0.1
            )
            
            response_text = response.choices[0].message.content.strip()
            logger.info("✅ OpenAI response generated successfully")
            return response_text
            
        except Exception as e:
            logger.error(f"❌ OpenAI generation failed: {e}")
            return self._generate_fallback_response(query, [])
    
    def _create_system_prompt(self, user_language: Optional[str]) -> str:
        """
        Create system prompt for OpenAI with clear instructions.
        
        Why this prompt design:
        - Sets clear role and behavior expectations
        - Emphasizes accuracy and source-based responses
        - Handles multilingual requirements
        - Prevents hallucination by emphasizing context-only responses
        """
        language_instruction = ""
        if user_language:
            if user_language.lower() == "gujarati":
                language_instruction = "Please respond in Gujarati (ગુજરાતી) when appropriate."
            elif user_language.lower() == "hindi":
                language_instruction = "Please respond in Hindi (हिन्दी) when appropriate."
            elif user_language.lower() == "hinglish":
                language_instruction = "Please respond in Hinglish (Hindi-English mix) when appropriate."
            else:
                language_instruction = "Please respond in English."
        
        return f"""You are Adhyatmik Intelligence AI, a knowledgeable assistant specializing in spiritual discourse from Akshar Sarjan Sabha sessions.

Your expertise covers:
- Akshar Sarjan Sabha spiritual sessions and teachings
- Swadhyay (self-study) sessions and discussions  
- Spiritual topics like meditation, devotion, wisdom, and self-improvement
- Teachings from spiritual leaders including Jignesh Kakadiya (Creator) and other speakers
- Topics ranging from practical life guidance to deep spiritual concepts

Your role:
- Provide detailed answers using the session transcripts provided in the context
- The sessions contain multilingual content (Gujarati, Hindi, English) - analyze all languages
- When users ask about "topics covered" or "what is discussed", provide comprehensive lists from the context
- Make connections between related spiritual concepts and practical applications
- Reference specific sessions, speakers, and dates when available
- Explain spiritual concepts in accessible terms

{language_instruction}

Key Guidelines:
- ALWAYS use the provided session context - it contains rich information about Akshar Sarjan Sabha
- When asked about topics, extract and list all topics, themes, and subjects from the context
- Connect related concepts (e.g., "thoughts management" with "Filter of Thoughts")
- Provide specific session references (speaker, date, topic) when possible
- If content is in Gujarati/Hindi, translate key points for English responses
- Be comprehensive - the sessions cover many diverse spiritual and practical topics
- Maintain a respectful, educational tone appropriate for spiritual content"""
    
    def _generate_fallback_response(self, query: str, retrieved_docs: List[Dict]) -> str:
        """
        Generate fallback response when OpenAI is not available.
        
        This provides basic functionality even without OpenAI API access.
        """
        if not retrieved_docs:
            return "I apologize, but I couldn't find relevant information in the session transcripts to answer your question. Please try rephrasing your question or ask about topics covered in the sessions."
        
        # Simple template-based response
        response_parts = [
            "Based on the session transcripts, here's what I found:",
            ""
        ]
        
        for i, doc in enumerate(retrieved_docs[:3], 1):  # Top 3 results
            speaker = doc['metadata'].get('speaker', 'Unknown')
            topic = doc['metadata'].get('topic', 'General discussion')
            content_preview = doc['content'][:200] + "..." if len(doc['content']) > 200 else doc['content']
            
            response_parts.append(f"{i}. From {speaker} on {topic}:")
            response_parts.append(f"   {content_preview}")
            response_parts.append("")
        
        response_parts.append("This information comes from the session transcripts in our database.")
        
        return "\n".join(response_parts)
    
    def _calculate_confidence(self, retrieved_docs: List[Dict], response_text: str) -> float:
        """
        Calculate confidence score for the response.
        
        Factors considered:
        - Number of retrieved documents
        - Average similarity score of retrieved documents
        - Length and quality of response
        - Presence of specific information
        """
        if not retrieved_docs:
            return 0.1
        
        # Base confidence from document similarity scores
        avg_similarity = sum(doc.get("similarity_score", 0) for doc in retrieved_docs) / len(retrieved_docs)
        
        # Adjust based on number of supporting documents
        doc_count_factor = min(len(retrieved_docs) / 3, 1.0)  # Normalize to 3 docs
        
        # Adjust based on response length (longer responses often indicate more context)
        response_length_factor = min(len(response_text) / 200, 1.0)  # Normalize to 200 chars
        
        # Combine factors
        confidence = (avg_similarity * 0.6) + (doc_count_factor * 0.2) + (response_length_factor * 0.2)
        
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def _create_no_context_response(self, query: str, user_language: Optional[str]) -> Dict[str, Any]:
        """Create response when no relevant documents are found."""
        return {
            "response": "I apologize, but I couldn't find relevant information in the session transcripts to answer your question. Please try rephrasing your question or ask about topics that might be covered in the spiritual and meditation sessions.",
            "language": user_language or "english",
            "confidence": 0.1,
            "sources": [],
            "processing_time": 0.1,
            "retrieved_docs_count": 0,
            "context_length": 0
        }
    
    def _create_error_response(self, error_message: str) -> Dict[str, Any]:
        """Create response when an error occurs."""
        return {
            "response": "I apologize, but I encountered an error while processing your question. Please try again later.",
            "language": "english",
            "confidence": 0.0,
            "sources": [],
            "processing_time": 0.0,
            "retrieved_docs_count": 0,
            "context_length": 0,
            "error": error_message
        }


# Global instance
rag_service = None


def get_rag_service(vector_db: VectorDatabaseService) -> RAGService:
    """
    Get the global RAG service instance.
    
    This ensures we have one RAG service shared across all API requests,
    which is efficient for resource usage and maintains consistency.
    """
    global rag_service
    if rag_service is None:
        rag_service = RAGService(vector_db)
    return rag_service


def initialize_rag_service(vector_db: VectorDatabaseService) -> RAGService:
    """
    Initialize the RAG service during application startup.
    """
    global rag_service
    rag_service = RAGService(vector_db)
    logger.info("🚀 RAG service initialized")
    return rag_service
