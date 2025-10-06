#!/usr/bin/env python3
"""
Test script for the Legal Document RAG Pipeline API
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_health():
    """Test the health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_collections():
    """Test the collections endpoint"""
    print("\nTesting collections endpoint...")
    response = requests.get(f"{BASE_URL}/collections")
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        print(f"Error: {response.text}")
    return response.status_code == 200

def test_query(question):
    """Test the query endpoint"""
    print(f"\nTesting query: '{question}'")
    
    payload = {"question": question}
    response = requests.post(
        f"{BASE_URL}/query",
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Answer: {result['answer'][:200]}...")
        print(f"Sources found: {len(result['sources'])}")
        for i, source in enumerate(result['sources'][:2]):
            print(f"  Source {i+1}: {source['document']} (page {source['page']}, confidence: {source['confidence']})")
    else:
        print(f"Error: {response.text}")
    
    return response.status_code == 200

def test_ingest(file_path):
    """Test document ingestion"""
    print(f"\nTesting document ingestion: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            files = {'file': f}
            response = requests.post(f"{BASE_URL}/ingest", files=files)
        
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Document: {result['document_name']}")
            print(f"Chunks processed: {result['chunks_processed']}")
        else:
            print(f"Error: {response.text}")
        
        return response.status_code == 200
    
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return False

def main():
    """Run all tests"""
    print("=== Legal Document RAG Pipeline API Tests ===\n")
    
    # Test basic endpoints
    health_ok = test_health()
    if not health_ok:
        print("❌ Health check failed. Make sure the server is running.")
        return
    
    collections_ok = test_collections()
    
    # Test sample queries
    test_queries = [
        "What are the termination procedures?",
        "How is confidential information protected?",
        "What are the payment terms?",
        "Who owns intellectual property?"
    ]
    
    print("\n=== Testing Sample Queries ===")
    for query in test_queries:
        test_query(query)
    
    print("\n=== Test Complete ===")
    print("✅ API is running successfully!")
    print("📝 To upload documents, use:")
    print("   curl -X POST 'http://localhost:8000/ingest' -F 'file=@your_document.pdf'")

if __name__ == "__main__":
    main()