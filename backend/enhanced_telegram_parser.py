#!/usr/bin/env python3
"""
Enhanced Telegram Chat Parser with Document Processing Pipeline
==============================================================

This enhanced parser combines the original Telegram parsing with the new
document processing pipeline for optimal RAG performance.

Features:
- All original Telegram parsing capabilities
- Integrated document processing pipeline
- Intelligent chunking and content enhancement
- Quality filtering and metadata enrichment
- Batch processing with progress tracking

Usage:
    python3 enhanced_telegram_parser.py --input chats/messages.html --process --upload
"""

import argparse
import logging
from typing import List, Dict, Any
from telegram_parser import TelegramChatParser
from services.document_processor import DocumentProcessor
from services.vector_db import VectorDatabaseService

logger = logging.getLogger(__name__)

class EnhancedTelegramParser(TelegramChatParser):
    """
    Enhanced Telegram parser with integrated document processing pipeline.
    
    Extends the original TelegramChatParser with:
    - Document processing pipeline integration
    - Intelligent chunking
    - Quality enhancement
    - Batch processing capabilities
    """
    
    def __init__(self):
        """Initialize enhanced parser with document processor."""
        super().__init__()
        self.document_processor = DocumentProcessor()
        self.processed_chunks = []
        
        logger.info("🚀 Enhanced Telegram Parser initialized")
    
    def parse_and_process(self, file_path: str, enable_processing: bool = True) -> Dict[str, Any]:
        """
        Parse Telegram HTML and process documents through the pipeline.
        
        Args:
            file_path: Path to the messages.html file
            enable_processing: Whether to run document processing pipeline
            
        Returns:
            Dictionary with parsing and processing results
        """
        logger.info(f"📖 Enhanced parsing of: {file_path}")
        
        # Step 1: Parse Telegram HTML (original functionality)
        sessions = self.parse_html_file(file_path)
        
        if not sessions:
            return {"sessions": [], "chunks": [], "stats": {}}
        
        # Step 2: Process documents if enabled
        if enable_processing:
            logger.info("🔄 Running document processing pipeline...")
            self.processed_chunks = self._process_sessions(sessions)
        else:
            logger.info("⏭️ Skipping document processing pipeline")
            self.processed_chunks = []
        
        # Step 3: Generate statistics
        stats = self._generate_stats(sessions, self.processed_chunks)
        
        return {
            "sessions": sessions,
            "chunks": self.processed_chunks,
            "stats": stats
        }
    
    def _process_sessions(self, sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process all sessions through the document processing pipeline.
        
        Args:
            sessions: List of parsed sessions
            
        Returns:
            List of processed document chunks
        """
        all_chunks = []
        
        logger.info(f"📄 Processing {len(sessions)} sessions through pipeline...")
        
        for i, session in enumerate(sessions, 1):
            try:
                logger.info(f"🔄 Processing session {i}/{len(sessions)}: {session['session_id']}")
                
                # Process this session
                chunks = self.document_processor.process_document(
                    content=session['content'],
                    document_id=session['session_id'],
                    metadata=session['metadata']
                )
                
                # Filter chunks by quality
                quality_chunks = [
                    chunk for chunk in chunks 
                    if chunk.get('quality_score', 0) >= 0.3  # Minimum quality threshold
                ]
                
                logger.info(f"✅ Session {session['session_id']}: {len(chunks)} chunks created, {len(quality_chunks)} passed quality filter")
                
                all_chunks.extend(quality_chunks)
                
            except Exception as e:
                logger.error(f"❌ Error processing session {session['session_id']}: {e}")
                continue
        
        logger.info(f"🎉 Document processing complete: {len(all_chunks)} high-quality chunks created")
        return all_chunks
    
    def _generate_stats(self, sessions: List[Dict], chunks: List[Dict]) -> Dict[str, Any]:
        """
        Generate comprehensive statistics about parsing and processing.
        
        Args:
            sessions: List of parsed sessions
            chunks: List of processed chunks
            
        Returns:
            Dictionary with detailed statistics
        """
        # Session statistics
        session_stats = {
            "total_sessions": len(sessions),
            "languages": {},
            "speakers": set(),
            "topics": set(),
            "date_range": {"earliest": None, "latest": None}
        }
        
        for session in sessions:
            # Language stats
            lang = session['language']
            session_stats["languages"][lang] = session_stats["languages"].get(lang, 0) + 1
            
            # Speaker stats
            session_stats["speakers"].update(session['speakers'])
            
            # Topic stats
            topic = session['metadata'].get('topic')
            if topic:
                session_stats["topics"].add(topic)
            
            # Date range
            session_date = session['metadata'].get('session_date')
            if session_date:
                if not session_stats["date_range"]["earliest"] or session_date < session_stats["date_range"]["earliest"]:
                    session_stats["date_range"]["earliest"] = session_date
                if not session_stats["date_range"]["latest"] or session_date > session_stats["date_range"]["latest"]:
                    session_stats["date_range"]["latest"] = session_date
        
        # Convert sets to lists for JSON serialization
        session_stats["speakers"] = list(session_stats["speakers"])
        session_stats["topics"] = list(session_stats["topics"])
        
        # Chunk statistics
        chunk_stats = {
            "total_chunks": len(chunks),
            "average_quality": sum(chunk.get('quality_score', 0) for chunk in chunks) / len(chunks) if chunks else 0,
            "language_distribution": {},
            "chunk_types": {},
            "quality_distribution": {"high": 0, "medium": 0, "low": 0}
        }
        
        for chunk in chunks:
            metadata = chunk.get('metadata', {})
            
            # Language distribution
            lang = metadata.get('primary_language', 'unknown')
            chunk_stats["language_distribution"][lang] = chunk_stats["language_distribution"].get(lang, 0) + 1
            
            # Chunk types
            chunk_type = metadata.get('chunk_type', 'unknown')
            chunk_stats["chunk_types"][chunk_type] = chunk_stats["chunk_types"].get(chunk_type, 0) + 1
            
            # Quality distribution
            quality = chunk.get('quality_score', 0)
            if quality >= 0.7:
                chunk_stats["quality_distribution"]["high"] += 1
            elif quality >= 0.4:
                chunk_stats["quality_distribution"]["medium"] += 1
            else:
                chunk_stats["quality_distribution"]["low"] += 1
        
        return {
            "sessions": session_stats,
            "chunks": chunk_stats,
            "processing_summary": {
                "total_documents_processed": len(sessions),
                "total_chunks_created": len(chunks),
                "average_chunks_per_session": len(chunks) / len(sessions) if sessions else 0,
                "quality_pass_rate": len([c for c in chunks if c.get('quality_score', 0) >= 0.3]) / len(chunks) if chunks else 0
            }
        }
    
    def upload_processed_chunks(self, upload_original: bool = False) -> Dict[str, Any]:
        """
        Upload processed chunks to the vector database.
        
        Args:
            upload_original: Whether to also upload original sessions
            
        Returns:
            Dictionary with upload results
        """
        if not self.processed_chunks:
            logger.warning("⚠️ No processed chunks to upload")
            return {"success": False, "message": "No processed chunks available"}
        
        try:
            logger.info("🚀 Uploading processed chunks to vector database...")
            
            # Initialize vector database
            vector_db = VectorDatabaseService()
            
            # Upload processed chunks to 'chunks' collection
            chunk_success_count = 0
            for chunk in self.processed_chunks:
                try:
                    success = vector_db.add_document(
                        content=chunk['content'],
                        document_id=chunk['chunk_id'],
                        metadata=chunk['metadata'],
                        collection_name="chunks"  # Use chunks collection for processed content
                    )
                    
                    if success:
                        chunk_success_count += 1
                    else:
                        logger.warning(f"⚠️ Failed to upload chunk: {chunk['chunk_id']}")
                        
                except Exception as e:
                    logger.error(f"❌ Error uploading chunk {chunk['chunk_id']}: {e}")
            
            # Optionally upload original sessions
            session_success_count = 0
            if upload_original and hasattr(self, 'sessions'):
                logger.info("📄 Also uploading original sessions...")
                for session in self.sessions:
                    try:
                        success = vector_db.add_document(
                            content=session['content'],
                            document_id=session['session_id'],
                            metadata=session['metadata'],
                            collection_name="transcripts"  # Use transcripts collection for original content
                        )
                        
                        if success:
                            session_success_count += 1
                            
                    except Exception as e:
                        logger.error(f"❌ Error uploading session {session['session_id']}: {e}")
            
            # Generate upload report
            upload_report = {
                "success": True,
                "chunks_uploaded": chunk_success_count,
                "chunks_total": len(self.processed_chunks),
                "chunks_success_rate": chunk_success_count / len(self.processed_chunks),
                "sessions_uploaded": session_success_count,
                "sessions_total": len(self.sessions) if hasattr(self, 'sessions') else 0,
                "message": f"Successfully uploaded {chunk_success_count}/{len(self.processed_chunks)} chunks"
            }
            
            if upload_original:
                upload_report["message"] += f" and {session_success_count}/{len(self.sessions)} sessions"
            
            logger.info(f"🎉 Upload complete: {upload_report['message']}")
            return upload_report
            
        except Exception as e:
            logger.error(f"❌ Error during upload: {e}")
            return {"success": False, "message": f"Upload failed: {str(e)}"}
    
    def export_processed_data(self, output_dir: str = "processed_output") -> Dict[str, str]:
        """
        Export processed data to files for analysis.
        
        Args:
            output_dir: Directory to save exported files
            
        Returns:
            Dictionary with paths to exported files
        """
        import json
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        exported_files = {}
        
        try:
            # Export processed chunks
            if self.processed_chunks:
                chunks_file = output_path / "processed_chunks.json"
                with open(chunks_file, 'w', encoding='utf-8') as f:
                    json.dump(self.processed_chunks, f, ensure_ascii=False, indent=2)
                exported_files["chunks"] = str(chunks_file)
                logger.info(f"📄 Exported {len(self.processed_chunks)} chunks to {chunks_file}")
            
            # Export original sessions
            if hasattr(self, 'sessions') and self.sessions:
                sessions_file = output_path / "original_sessions.json"
                with open(sessions_file, 'w', encoding='utf-8') as f:
                    json.dump(self.sessions, f, ensure_ascii=False, indent=2)
                exported_files["sessions"] = str(sessions_file)
                logger.info(f"📄 Exported {len(self.sessions)} sessions to {sessions_file}")
            
            # Export statistics
            if hasattr(self, 'stats'):
                stats_file = output_path / "processing_stats.json"
                with open(stats_file, 'w', encoding='utf-8') as f:
                    json.dump(self.stats, f, ensure_ascii=False, indent=2)
                exported_files["stats"] = str(stats_file)
                logger.info(f"📊 Exported statistics to {stats_file}")
            
            return exported_files
            
        except Exception as e:
            logger.error(f"❌ Error exporting data: {e}")
            return {}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(description='Enhanced Telegram chat parser with document processing')
    parser.add_argument('--input', '-i', required=True, help='Path to messages.html file')
    parser.add_argument('--process', action='store_true', help='Enable document processing pipeline')
    parser.add_argument('--upload', action='store_true', help='Upload to vector database')
    parser.add_argument('--upload-original', action='store_true', help='Also upload original sessions')
    parser.add_argument('--export', help='Export processed data to directory')
    parser.add_argument('--preview', action='store_true', help='Preview processing results')
    
    args = parser.parse_args()
    
    # Initialize enhanced parser
    enhanced_parser = EnhancedTelegramParser()
    
    # Parse and process
    results = enhanced_parser.parse_and_process(args.input, enable_processing=args.process)
    
    # Store results for other operations
    enhanced_parser.sessions = results["sessions"]
    enhanced_parser.stats = results["stats"]
    
    # Preview results if requested
    if args.preview:
        logger.info("👀 Processing Results Preview:")
        print(f"\n📊 Statistics:")
        print(f"   Sessions parsed: {results['stats']['sessions']['total_sessions']}")
        print(f"   Chunks created: {results['stats']['chunks']['total_chunks']}")
        print(f"   Average quality: {results['stats']['chunks']['average_quality']:.2f}")
        print(f"   Languages: {list(results['stats']['sessions']['languages'].keys())}")
        
        if results["chunks"]:
            print(f"\n📄 Sample processed chunk:")
            sample_chunk = results["chunks"][0]
            print(f"   ID: {sample_chunk['chunk_id']}")
            print(f"   Quality: {sample_chunk['quality_score']:.2f}")
            print(f"   Language: {sample_chunk['metadata']['primary_language']}")
            print(f"   Content preview: {sample_chunk['content'][:200]}...")
    
    # Export if requested
    if args.export:
        exported = enhanced_parser.export_processed_data(args.export)
        logger.info(f"📁 Exported files: {list(exported.keys())}")
    
    # Upload if requested
    if args.upload:
        upload_result = enhanced_parser.upload_processed_chunks(upload_original=args.upload_original)
        if upload_result["success"]:
            logger.info("🎉 Upload completed successfully!")
        else:
            logger.error("❌ Upload failed!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    main()
