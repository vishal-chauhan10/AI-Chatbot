#!/usr/bin/env python3
"""
Telegram Chat Parser for Spiritual Discourse Content
===================================================

This parser extracts spiritual discourse content from exported Telegram chats
and structures it for RAG (Retrieval-Augmented Generation) systems.

Features:
- Extracts session information (dates, venues, topics, sabha names)
- Identifies speakers and content creators
- Processes multilingual content (Gujarati, English, Hindi)
- Structures data with rich metadata for semantic search
- Bulk uploads to ChromaDB for RAG integration

Usage:
    python3 telegram_parser.py --input chats/messages.html --upload
"""

import re
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from bs4 import BeautifulSoup
import logging

# Import our RAG services
from services.vector_db import VectorDatabaseService
from config import get_settings

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TelegramChatParser:
    """
    Parser for Telegram exported chat HTML files containing spiritual discourse content.
    
    Extracts:
    - Session metadata (dates, venues, topics, sabha names)
    - Speaker information and content
    - Message threading and continuity
    - Multilingual content processing
    """
    
    def __init__(self):
        self.sessions = []
        self.current_session = None
        self.message_buffer = []
        
        # Known creators/speakers (can be expanded)
        self.known_creators = {
            "Jignesh Kakadiya": "Creator",
            "જિગ્નેશ કાકડિયા": "Creator",
            # Add more known speakers here
        }
        
        # Session patterns
        self.session_patterns = {
            'sabha_name': re.compile(r'(.*?)\s*Sabha', re.IGNORECASE),
            'date': re.compile(r'Date\s*:?\s*(\d{1,2})\s*-\s*(\w+)\s*-\s*(\d{4})\s*\((\w+)\)', re.IGNORECASE),
            'place': re.compile(r'Place\s*[-:]?\s*(.+?)(?:\n|$)', re.IGNORECASE),
            'topic': re.compile(r'Topic\s*:?\s*(.+?)(?:\n|$)', re.IGNORECASE)
        }
        
        logger.info("🎭 Telegram Chat Parser initialized")
    
    def parse_html_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Parse the exported Telegram HTML file and extract structured content.
        
        Args:
            file_path: Path to the exported messages.html file
            
        Returns:
            List of structured session documents
        """
        logger.info(f"📖 Parsing Telegram chat file: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            soup = BeautifulSoup(content, 'html.parser')
            
            # Find all message elements
            messages = soup.find_all('div', class_='message')
            logger.info(f"📨 Found {len(messages)} messages to process")
            
            for message in messages:
                self._process_message(message)
            
            # Process any remaining buffered messages
            if self.message_buffer:
                self._finalize_current_session()
            
            logger.info(f"✅ Extracted {len(self.sessions)} sessions")
            return self.sessions
            
        except Exception as e:
            logger.error(f"❌ Error parsing HTML file: {e}")
            return []
    
    def _process_message(self, message_element) -> None:
        """Process a single message element from the HTML."""
        try:
            # Extract message metadata
            sender = self._extract_sender(message_element)
            timestamp = self._extract_timestamp(message_element)
            content = self._extract_content(message_element)
            
            if not content:
                return
            
            # Check if this message starts a new session
            session_info = self._detect_session_start(content)
            
            if session_info:
                # Finalize previous session if exists
                if self.message_buffer:
                    self._finalize_current_session()
                
                # Start new session
                self._start_new_session(session_info, sender, timestamp)
            
            # Add message to current session buffer
            if content.strip():
                self.message_buffer.append({
                    'sender': sender,
                    'timestamp': timestamp,
                    'content': content.strip(),
                    'speaker_type': self._classify_speaker(sender)
                })
                
        except Exception as e:
            logger.warning(f"⚠️ Error processing message: {e}")
    
    def _extract_sender(self, message_element) -> str:
        """Extract sender name from message element."""
        sender_elem = message_element.find('div', class_='from_name')
        if sender_elem:
            return sender_elem.get_text(strip=True)
        return "Unknown"
    
    def _extract_timestamp(self, message_element) -> Optional[str]:
        """Extract timestamp from message element."""
        time_elem = message_element.find('div', class_='date')
        if time_elem:
            return time_elem.get('title') or time_elem.get_text(strip=True)
        return None
    
    def _extract_content(self, message_element) -> str:
        """Extract message content, handling various HTML structures."""
        content_elem = message_element.find('div', class_='text')
        if content_elem:
            # Handle different content types (text, media, etc.)
            text_content = content_elem.get_text(separator='\n', strip=True)
            return text_content
        return ""
    
    def _detect_session_start(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Detect if a message indicates the start of a new session.
        
        Looks for patterns like:
        - Sabha names
        - Date information
        - Place information
        - Topic information
        """
        session_info = {}
        
        # Check for sabha name
        sabha_match = self.session_patterns['sabha_name'].search(content)
        if sabha_match:
            session_info['sabha_name'] = sabha_match.group(1).strip()
        
        # Check for date
        date_match = self.session_patterns['date'].search(content)
        if date_match:
            day, month, year, weekday = date_match.groups()
            session_info['date'] = {
                'day': int(day),
                'month': month,
                'year': int(year),
                'weekday': weekday,
                'raw': date_match.group(0)
            }
        
        # Check for place
        place_match = self.session_patterns['place'].search(content)
        if place_match:
            session_info['place'] = place_match.group(1).strip()
        
        # Check for topic
        topic_match = self.session_patterns['topic'].search(content)
        if topic_match:
            session_info['topic'] = topic_match.group(1).strip()
        
        # Return session info if we found key indicators
        if len(session_info) >= 2:  # At least 2 pieces of session info
            return session_info
        
        return None
    
    def _start_new_session(self, session_info: Dict[str, Any], sender: str, timestamp: str) -> None:
        """Start tracking a new session."""
        self.current_session = {
            'session_info': session_info,
            'started_by': sender,
            'start_timestamp': timestamp,
            'messages': []
        }
        
        logger.info(f"🎭 New session detected: {session_info.get('sabha_name', 'Unknown Sabha')}")
        if 'topic' in session_info:
            logger.info(f"📝 Topic: {session_info['topic']}")
    
    def _finalize_current_session(self) -> None:
        """Finalize the current session and add it to sessions list."""
        if not self.current_session or not self.message_buffer:
            return
        
        # Combine all messages into session content
        session_content = self._combine_messages(self.message_buffer)
        
        # Create structured session document
        session_doc = {
            'session_id': self._generate_session_id(),
            'metadata': self._create_session_metadata(),
            'content': session_content,
            'message_count': len(self.message_buffer),
            'speakers': list(set(msg['sender'] for msg in self.message_buffer)),
            'language': self._detect_primary_language(session_content)
        }
        
        self.sessions.append(session_doc)
        
        # Reset for next session
        self.message_buffer = []
        self.current_session = None
    
    def _combine_messages(self, messages: List[Dict[str, Any]]) -> str:
        """Combine messages into coherent session content."""
        content_parts = []
        
        current_speaker = None
        speaker_content = []
        
        for msg in messages:
            if msg['sender'] != current_speaker:
                # New speaker, finalize previous speaker's content
                if current_speaker and speaker_content:
                    speaker_text = ' '.join(speaker_content)
                    if len(speaker_text.strip()) > 10:  # Only include substantial content
                        content_parts.append(f"{current_speaker}: {speaker_text}")
                
                current_speaker = msg['sender']
                speaker_content = [msg['content']]
            else:
                speaker_content.append(msg['content'])
        
        # Add final speaker's content
        if current_speaker and speaker_content:
            speaker_text = ' '.join(speaker_content)
            if len(speaker_text.strip()) > 10:
                content_parts.append(f"{current_speaker}: {speaker_text}")
        
        return '\n\n'.join(content_parts)
    
    def _create_session_metadata(self) -> Dict[str, Any]:
        """Create rich metadata for the session."""
        if not self.current_session:
            return {}
        
        session_info = self.current_session['session_info']
        
        metadata = {
            'content_type': 'telegram_session',
            'source': 'telegram_export',
            'processed_at': datetime.utcnow().isoformat(),
        }
        
        # Add session-specific metadata
        if 'sabha_name' in session_info:
            metadata['sabha_name'] = session_info['sabha_name']
            metadata['event_type'] = 'spiritual_sabha'
        
        if 'date' in session_info:
            date_info = session_info['date']
            metadata['session_date'] = f"{date_info['year']}-{self._month_to_number(date_info['month']):02d}-{date_info['day']:02d}"
            metadata['weekday'] = date_info['weekday']
        
        if 'place' in session_info:
            metadata['venue'] = session_info['place']
        
        if 'topic' in session_info:
            metadata['topic'] = session_info['topic']
        
        # Add speaker information
        creators = [msg['sender'] for msg in self.message_buffer if msg['speaker_type'] == 'Creator']
        if creators:
            metadata['primary_speaker'] = creators[0]
            metadata['speaker_type'] = 'spiritual_teacher'
        
        return metadata
    
    def _generate_session_id(self) -> str:
        """Generate a unique session ID."""
        if not self.current_session:
            return f"telegram_session_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        
        session_info = self.current_session['session_info']
        
        # Create ID from sabha name and date
        sabha = session_info.get('sabha_name', 'unknown_sabha').lower().replace(' ', '_')
        
        if 'date' in session_info:
            date_info = session_info['date']
            date_str = f"{date_info['year']}{date_info['month'][:3].lower()}{date_info['day']:02d}"
            return f"{sabha}_{date_str}"
        
        return f"{sabha}_{datetime.utcnow().strftime('%Y%m%d')}"
    
    def _classify_speaker(self, sender: str) -> str:
        """Classify speaker type based on known creators."""
        if sender in self.known_creators:
            return self.known_creators[sender]
        
        # Check for partial matches or variations
        for known_name, role in self.known_creators.items():
            if known_name.lower() in sender.lower() or sender.lower() in known_name.lower():
                return role
        
        return "Participant"
    
    def _detect_primary_language(self, content: str) -> str:
        """Detect the primary language of the session content."""
        # Simple heuristic based on character sets
        gujarati_chars = len(re.findall(r'[\u0A80-\u0AFF]', content))
        hindi_chars = len(re.findall(r'[\u0900-\u097F]', content))
        total_chars = len(content)
        
        if gujarati_chars > total_chars * 0.3:
            return "gujarati"
        elif hindi_chars > total_chars * 0.3:
            return "hindi"
        else:
            return "english"
    
    def _month_to_number(self, month_name: str) -> int:
        """Convert month name to number."""
        months = {
            'jan': 1, 'january': 1,
            'feb': 2, 'february': 2,
            'mar': 3, 'march': 3,
            'apr': 4, 'april': 4,
            'may': 5,
            'jun': 6, 'june': 6,
            'jul': 7, 'july': 7,
            'aug': 8, 'august': 8,
            'sep': 9, 'sept': 9, 'september': 9,
            'oct': 10, 'october': 10,
            'nov': 11, 'november': 11,
            'dec': 12, 'december': 12
        }
        return months.get(month_name.lower(), 1)
    
    def export_to_json(self, output_path: str) -> None:
        """Export parsed sessions to JSON file."""
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(self.sessions, f, ensure_ascii=False, indent=2)
            logger.info(f"📄 Exported {len(self.sessions)} sessions to {output_path}")
        except Exception as e:
            logger.error(f"❌ Error exporting to JSON: {e}")
    
    def upload_to_vector_db(self) -> bool:
        """Upload parsed sessions to ChromaDB for RAG."""
        try:
            logger.info("🚀 Uploading sessions to vector database...")
            
            # Initialize vector database
            vector_db = VectorDatabaseService()
            
            success_count = 0
            for session in self.sessions:
                try:
                    # Upload to transcripts collection
                    success = vector_db.add_document(
                        content=session['content'],
                        document_id=session['session_id'],
                        metadata=session['metadata'],
                        collection_name="transcripts"
                    )
                    
                    if success:
                        success_count += 1
                        logger.info(f"✅ Uploaded: {session['session_id']}")
                    else:
                        logger.warning(f"⚠️ Failed to upload: {session['session_id']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error uploading {session['session_id']}: {e}")
            
            logger.info(f"🎉 Successfully uploaded {success_count}/{len(self.sessions)} sessions")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"❌ Error during vector database upload: {e}")
            return False


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Parse Telegram chat export for RAG system')
    parser.add_argument('--input', '-i', required=True, help='Path to messages.html file')
    parser.add_argument('--output', '-o', help='Output JSON file path')
    parser.add_argument('--upload', action='store_true', help='Upload to vector database')
    parser.add_argument('--preview', action='store_true', help='Preview first few sessions')
    
    args = parser.parse_args()
    
    # Initialize parser
    telegram_parser = TelegramChatParser()
    
    # Parse the HTML file
    sessions = telegram_parser.parse_html_file(args.input)
    
    if not sessions:
        logger.error("❌ No sessions found or parsing failed")
        return
    
    # Preview sessions if requested
    if args.preview:
        logger.info("👀 Preview of parsed sessions:")
        for i, session in enumerate(sessions[:3]):
            print(f"\n📋 Session {i+1}: {session['session_id']}")
            print(f"   Topic: {session['metadata'].get('topic', 'Unknown')}")
            print(f"   Date: {session['metadata'].get('session_date', 'Unknown')}")
            print(f"   Speakers: {', '.join(session['speakers'])}")
            print(f"   Content preview: {session['content'][:200]}...")
    
    # Export to JSON if requested
    if args.output:
        telegram_parser.export_to_json(args.output)
    
    # Upload to vector database if requested
    if args.upload:
        success = telegram_parser.upload_to_vector_db()
        if success:
            logger.info("🎉 Upload completed successfully!")
        else:
            logger.error("❌ Upload failed!")


if __name__ == "__main__":
    main()
