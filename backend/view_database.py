#!/usr/bin/env python3
"""
ChromaDB Database Viewer
========================

This script provides a simple way to view and explore your ChromaDB database
containing the Gujarati spiritual discourse content.

Usage:
    python3 view_database.py

Features:
- List all collections
- View documents in each collection
- Search functionality
- Display metadata and embeddings info
"""

import chromadb
from pprint import pprint
import json
from datetime import datetime

def main():
    print("🔍 ChromaDB Database Viewer")
    print("=" * 50)
    
    try:
        # Connect to your database
        client = chromadb.PersistentClient(path="./chroma_db")
        print("✅ Connected to ChromaDB successfully!")
        
        # List all collections
        collections = client.list_collections()
        print(f"\n📚 Found {len(collections)} collections:")
        
        for i, collection in enumerate(collections, 1):
            print(f"{i}. {collection.name}")
            print(f"   Metadata: {collection.metadata}")
        
        # Get detailed info for each collection
        for collection in collections:
            print(f"\n" + "="*60)
            print(f"📖 Collection: {collection.name}")
            print("="*60)
            
            # Get all documents in this collection
            results = collection.get()
            
            print(f"Total documents: {len(results['ids'])}")
            
            if len(results['ids']) > 0:
                print("\n📄 Documents:")
                print("-" * 40)
                
                for i in range(len(results['ids'])):
                    doc_id = results['ids'][i]
                    content = results['documents'][i] if results['documents'] else "No content"
                    metadata = results['metadatas'][i] if results['metadatas'] else {}
                    
                    print(f"\n{i+1}. Document ID: {doc_id}")
                    print(f"   Speaker: {metadata.get('speaker', 'Unknown')}")
                    print(f"   Topic: {metadata.get('topic', 'Unknown')}")
                    print(f"   Language: {metadata.get('language', 'Unknown')}")
                    print(f"   Date: {metadata.get('session_date', 'Unknown')}")
                    
                    # Show first 200 characters of content
                    if content:
                        content_preview = content[:200] + "..." if len(content) > 200 else content
                        print(f"   Content: {content_preview}")
                    
                    print(f"   Full Metadata: {json.dumps(metadata, indent=2, ensure_ascii=False)}")
        
        # Interactive search
        print(f"\n" + "="*60)
        print("🔍 Interactive Search")
        print("="*60)
        
        while True:
            query = input("\nEnter search query (or 'quit' to exit): ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                break
            
            if not query:
                continue
            
            print(f"\nSearching for: '{query}'")
            print("-" * 40)
            
            # Search in transcripts collection
            try:
                transcripts = client.get_collection("session_transcripts")
                search_results = transcripts.query(
                    query_texts=[query],
                    n_results=3
                )
                
                if search_results['ids'][0]:  # Check if we have results
                    for i in range(len(search_results['ids'][0])):
                        doc_id = search_results['ids'][0][i]
                        content = search_results['documents'][0][i]
                        metadata = search_results['metadatas'][0][i]
                        distance = search_results['distances'][0][i] if search_results.get('distances') else None
                        
                        similarity = 1 - distance if distance is not None else "Unknown"
                        
                        print(f"\n📄 Result {i+1}:")
                        print(f"   ID: {doc_id}")
                        print(f"   Similarity: {similarity}")
                        print(f"   Speaker: {metadata.get('speaker', 'Unknown')}")
                        print(f"   Topic: {metadata.get('topic', 'Unknown')}")
                        
                        content_preview = content[:300] + "..." if len(content) > 300 else content
                        print(f"   Content: {content_preview}")
                else:
                    print("No results found.")
                    
            except Exception as e:
                print(f"Search error: {e}")
    
    except Exception as e:
        print(f"❌ Error connecting to database: {e}")
        print("\nMake sure:")
        print("1. You're in the backend directory")
        print("2. The chroma_db folder exists")
        print("3. Your FastAPI server has been run at least once")

if __name__ == "__main__":
    main()
