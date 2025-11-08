"""
AI-Powered Translation Service
=============================

This service uses OpenAI to automatically translate names and terms between 
English and Gujarati, eliminating the need for hard-coded name mappings.

Key Features:
- Automatic name variant generation using GPT-4o
- Intelligent caching to save API tokens
- Fallback handling for API failures
- Support for spiritual/religious terms and titles
"""

import logging
import json
from typing import List, Dict, Any, Optional
import openai
from datetime import datetime, timedelta
from config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()


class TranslationService:
    """
    AI-powered translation service for names and terms.
    Uses OpenAI to automatically translate between English and Gujarati.
    """
    
    def __init__(self):
        """Initialize the translation service with OpenAI client and cache."""
        self.openai_client = None
        self._initialize_openai()
        
        # Translation cache with timestamps for expiry
        self._translation_cache: Dict[str, Dict[str, Any]] = {}
        self.cache_expiry_hours = 24  # Cache expires after 24 hours
        
        logger.info("✅ Translation service initialized")
    
    def _initialize_openai(self):
        """Initialize OpenAI client with API key from environment."""
        try:
            if settings.openai_api_key:
                self.openai_client = openai.OpenAI(api_key=settings.openai_api_key)
                logger.info("✅ OpenAI client initialized for translation service")
            else:
                logger.warning("⚠️ OpenAI API key not found. Translation service will be limited.")
                self.openai_client = None
        except Exception as e:
            logger.error(f"❌ Failed to initialize OpenAI client for translation: {e}")
            self.openai_client = None
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if a cache entry is still valid (not expired)."""
        if 'timestamp' not in cache_entry:
            return False
        
        cache_time = datetime.fromisoformat(cache_entry['timestamp'])
        expiry_time = cache_time + timedelta(hours=self.cache_expiry_hours)
        return datetime.now() < expiry_time
    
    async def get_name_variants(self, search_term: str) -> List[str]:
        """
        Get all possible variants of a name/term using AI translation.
        
        Args:
            search_term: The name or term to get variants for
            
        Returns:
            List of all possible variants (English, Gujarati, transliterations)
        """
        if not search_term or not search_term.strip():
            return []
        
        # Normalize the search term for caching
        cache_key = search_term.lower().strip()
        
        # Check cache first
        if cache_key in self._translation_cache:
            cache_entry = self._translation_cache[cache_key]
            if self._is_cache_valid(cache_entry):
                logger.info(f"📋 Using cached variants for '{search_term}': {cache_entry['variants']}")
                return cache_entry['variants']
            else:
                # Remove expired cache entry
                del self._translation_cache[cache_key]
        
        # If no OpenAI client, return original term
        if not self.openai_client:
            logger.warning(f"⚠️ No OpenAI client available, returning original term: {search_term}")
            return [cache_key]
        
        try:
            logger.info(f"🤖 Generating AI variants for: '{search_term}'")
            
            # Create AI prompt for name translation
            prompt = f"""
You are a multilingual name translation expert specializing in English and Gujarati.

Given the name/term: "{search_term}"

Please provide ALL possible variants including:
1. Original term (as provided)
2. Gujarati script version (if applicable)
3. English transliterations and romanizations
4. Common variations, nicknames, and abbreviations
5. Formal and informal versions
6. With and without titles (Bhai, Swami, Ji, etc.)

For spiritual/religious names, include common titles and variations.
For common words, include both singular and plural forms.

Return ONLY a JSON array of strings, no explanations or additional text:
["variant1", "variant2", "variant3", ...]

Examples:
- Input: "Jignesh" → ["jignesh", "જિગ્નેશ", "jignesh kakadiya", "જિગ્નેશ કાકડિયા"]
- Input: "Swami" → ["swami", "સ્વામી", "swamiji", "સ્વામીજી"]
- Input: "Prabhudas" → ["prabhudas", "પ્રભુદાસ", "prabhudas bhai", "પ્રભુદાસભાઈ", "prabhudasbhai"]
"""

            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system", 
                        "content": "You are a multilingual translation expert. Return only valid JSON arrays with lowercase variants."
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.2,  # Low temperature for consistent translations
                top_p=0.9
            )
            
            # Parse the JSON response
            variants_text = response.choices[0].message.content.strip()
            
            # Clean up the response (remove code blocks if present)
            if variants_text.startswith('```'):
                variants_text = variants_text.split('\n', 1)[1]
            if variants_text.endswith('```'):
                variants_text = variants_text.rsplit('\n', 1)[0]
            
            variants = json.loads(variants_text)
            
            # Ensure we have a list
            if not isinstance(variants, list):
                raise ValueError("AI response is not a list")
            
            # Add original term and normalize
            all_variants = [cache_key]  # Always include the original search term
            for variant in variants:
                if variant and isinstance(variant, str) and variant.strip():
                    normalized_variant = variant.strip().lower()
                    if normalized_variant not in all_variants:
                        all_variants.append(normalized_variant)
            
            # Remove empty strings and duplicates while preserving order
            unique_variants = []
            seen = set()
            for variant in all_variants:
                if variant and variant not in seen:
                    unique_variants.append(variant)
                    seen.add(variant)
            
            # Cache the result with timestamp
            cache_entry = {
                'variants': unique_variants,
                'timestamp': datetime.now().isoformat(),
                'original_term': search_term
            }
            self._translation_cache[cache_key] = cache_entry
            
            logger.info(f"✅ Generated {len(unique_variants)} variants for '{search_term}': {unique_variants}")
            return unique_variants
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ Failed to parse AI response as JSON for '{search_term}': {e}")
            return [cache_key]
        except Exception as e:
            logger.error(f"❌ Translation failed for '{search_term}': {e}")
            return [cache_key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get statistics about the translation cache."""
        total_entries = len(self._translation_cache)
        valid_entries = sum(1 for entry in self._translation_cache.values() if self._is_cache_valid(entry))
        
        return {
            'total_cached_terms': total_entries,
            'valid_cached_terms': valid_entries,
            'expired_cached_terms': total_entries - valid_entries,
            'cache_hit_rate': f"{(valid_entries / max(total_entries, 1)) * 100:.1f}%"
        }
    
    def clear_expired_cache(self):
        """Remove expired entries from the cache."""
        expired_keys = [
            key for key, entry in self._translation_cache.items()
            if not self._is_cache_valid(entry)
        ]
        
        for key in expired_keys:
            del self._translation_cache[key]
        
        if expired_keys:
            logger.info(f"🧹 Cleared {len(expired_keys)} expired cache entries")


# Global instance
_translation_service: Optional[TranslationService] = None


def get_translation_service() -> TranslationService:
    """
    Get the global translation service instance.
    
    This ensures we have one translation service shared across all API requests,
    which is efficient for caching and resource usage.
    """
    global _translation_service
    if _translation_service is None:
        _translation_service = TranslationService()
    return _translation_service


def initialize_translation_service() -> TranslationService:
    """
    Initialize and return the translation service.
    
    This is called during application startup to ensure the service is ready.
    """
    global _translation_service
    _translation_service = TranslationService()
    return _translation_service
