"""
Enhanced RAG service with LLM synthesis and query understanding
Using Groq API for fast open-source LLM inference
"""
from langchain_groq import ChatGroq
from .rag_service import rag_service
from typing import List, Dict, Any
import logging
from datetime import datetime
import os
import json
import re

logger = logging.getLogger(__name__)

class QueryIntentAnalyzer:
    """Analyze query intent for better RAG retrieval"""
    
    @staticmethod
    def analyze_query_intent(query: str) -> Dict[str, Any]:
        """Understand what the user really wants"""
        
        analysis = {
            "query_type": "general",
            "scope": "single",
            "comparison_intent": False,
            "cross_document": False,
            "suggested_k": 5,
            "expansion_keywords": [],
            "intent_confidence": 0.5
        }
        
        query_lower = query.lower()
        
        # Detect comparison intent
        comparison_keywords = ["compare", "contrast", "difference", "differences", "across", "between", "all", "various", "different"]
        found_comparison = [word for word in comparison_keywords if word in query_lower]
        if found_comparison:
            analysis["comparison_intent"] = True
            analysis["cross_document"] = True
            analysis["suggested_k"] = 15
            analysis["expansion_keywords"] = found_comparison
            analysis["intent_confidence"] = 0.8
        
        # Detect cross-document scope
        cross_doc_keywords = ["all contracts", "all documents", "every contract", "across contracts", "multiple"]
        if any(keyword in query_lower for keyword in cross_doc_keywords):
            analysis["cross_document"] = True
            analysis["scope"] = "cross_document"
            analysis["suggested_k"] = max(analysis["suggested_k"], 12)
        
        # Detect specific topics
        topic_keywords = {
            "termination": ["termination", "terminate", "end", "cancel", "dissolution"],
            "payment": ["payment", "pay", "salary", "compensation", "fee", "cost", "invoice"],
            "confidentiality": ["confidential", "nda", "non-disclosure", "secret", "proprietary"],
            "ip": ["intellectual property", "copyright", "patent", "ownership", "work product"],
            "liability": ["liability", "damages", "responsible", "accountable", "indemnification"]
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in query_lower for keyword in keywords):
                analysis["query_type"] = topic
                break
        
        # Generate expansion queries for cross-document searches
        if analysis["cross_document"]:
            analysis["expansion_queries"] = QueryIntentAnalyzer.generate_expansion_queries(query, analysis["query_type"])
        
        return analysis
    
    @staticmethod
    def generate_expansion_queries(original_query: str, topic: str) -> List[str]:
        """Generate query variations for better coverage"""
        
        base_terms = {
            "termination": [
                "termination procedures",
                "notice requirements",
                "immediate termination",
                "termination for cause",
                "contract end",
                "cancellation terms"
            ],
            "payment": [
                "payment terms",
                "salary details",
                "compensation structure", 
                "fee schedule",
                "invoice terms",
                "payment obligations"
            ],
            "confidentiality": [
                "confidential information",
                "non-disclosure obligations",
                "proprietary information",
                "confidentiality terms",
                "information protection"
            ],
            "ip": [
                "intellectual property",
                "work product ownership",
                "copyright assignment",
                "invention rights",
                "IP ownership"
            ],
            "general": [original_query]
        }
        
        return base_terms.get(topic, [original_query])


class EnhancedRAGService:
    """Enhanced RAG with LLM synthesis and query understanding"""
    
    def __init__(self, groq_api_key: str = None):
        """Initialize enhanced RAG service"""
        self.base_rag = rag_service
        self.intent_analyzer = QueryIntentAnalyzer()
        
        # Initialize Groq LLM for synthesis
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY", "demo-key")
        self.synthesis_llm = self.setup_groq_llm()
        
    def setup_groq_llm(self):
        """Setup Groq LLM for synthesis"""
        try:
            if self.groq_api_key == "demo-key":
                logger.warning("Using demo Groq API key - get your free key at https://console.groq.com/")
                return None
            
            llm = ChatGroq(
                groq_api_key=self.groq_api_key,
                model_name="llama-3.1-8b-instant",  # Fast and capable
                temperature=0.1,
                max_tokens=1024
            )
            
            logger.info("Groq LLM initialized successfully")
            return llm
            
        except Exception as e:
            logger.error(f"Failed to initialize Groq LLM: {e}")
            return None
    
    def analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Analyze query intent and return analysis"""
        try:
            analysis = self.intent_analyzer.analyze_query_intent(query)
            
            return {
                "query": query,
                "analysis": analysis,
                "recommendations": {
                    "suggested_approach": "cross_document" if analysis["cross_document"] else "standard",
                    "suggested_k": analysis["suggested_k"],
                    "use_expansion": analysis["comparison_intent"],
                    "llm_synthesis_recommended": analysis["cross_document"] or analysis["comparison_intent"]
                },
                "expansion_queries": analysis.get("expansion_queries", []),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing query intent: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def query_with_intent_understanding(self, query: str, k: int = None) -> Dict[str, Any]:
        """Enhanced query using intent understanding"""
        try:
            # Step 1: Analyze intent
            intent_result = self.analyze_query_intent(query)
            analysis = intent_result.get("analysis", {})
            
            # Step 2: Adjust retrieval parameters
            optimal_k = k or analysis.get("suggested_k", 5)
            
            # Step 3: Perform retrieval
            if analysis.get("cross_document", False):
                # Multi-query approach for cross-document searches
                all_results = []
                
                # Original query
                results = self.base_rag.query_documents(query, k=optimal_k)
                all_results.extend(results)
                
                # Expansion queries if available
                for exp_query in analysis.get("expansion_queries", [])[:3]:  # Limit to 3 expansions
                    exp_results = self.base_rag.query_documents(exp_query, k=5)
                    all_results.extend(exp_results)
                
                # Deduplicate and re-rank
                final_results = self.deduplicate_results(all_results)
            else:
                # Standard single query
                final_results = self.base_rag.query_documents(query, k=optimal_k)
            
            # Step 4: Format response
            return {
                "answer": self.format_intent_based_response(query, final_results, analysis),
                "intent_analysis": analysis,
                "approach_used": "intent_understanding",
                "sources": final_results[:8],  # Show top 8 sources
                "query_time": datetime.now().isoformat(),
                "total_results": len(final_results)
            }
            
        except Exception as e:
            logger.error(f"Error in intent-based query: {e}")
            return {
                "error": str(e),
                "fallback_used": True,
                "timestamp": datetime.now().isoformat()
            }
    
    def query_with_llm_synthesis(self, query: str, k: int = 10) -> Dict[str, Any]:
        """Enhanced query with LLM synthesis"""
        try:
            # Step 1: Get comprehensive search results
            search_results = self.base_rag.query_documents(query, k=k)
            
            if not search_results:
                return {
                    "answer": "No relevant documents found for your query.",
                    "approach_used": "llm_synthesis",
                    "synthesis_used": False,
                    "sources": [],
                    "query_time": datetime.now().isoformat(),
                    "total_results": 0
                }
            
            # Step 2: LLM synthesis if available
            if self.synthesis_llm:
                synthesized_answer = self.synthesize_with_llm(query, search_results)
                return {
                    "answer": synthesized_answer,
                    "approach_used": "llm_synthesis",
                    "synthesis_used": True,
                    "model": "llama-3.1-8b-instant",
                    "sources": search_results[:5],
                    "query_time": datetime.now().isoformat(),
                    "total_results": len(search_results)
                }
            else:
                # Fallback to enhanced formatting
                return {
                    "answer": self.create_enhanced_answer(query, search_results),
                    "approach_used": "enhanced_formatting",
                    "synthesis_used": False,
                    "note": "LLM synthesis unavailable - using enhanced formatting",
                    "sources": search_results[:5],
                    "query_time": datetime.now().isoformat(),
                    "total_results": len(search_results)
                }
                
        except Exception as e:
            logger.error(f"Error in LLM synthesis query: {e}")
            return {
                "error": str(e),
                "approach_used": "error_fallback",
                "timestamp": datetime.now().isoformat()
            }
    
    def synthesize_with_llm(self, query: str, results: List[Dict]) -> str:
        """Use Groq LLM to synthesize comprehensive response"""
        try:
            # Prepare context
            context = self.prepare_llm_context(results)
            
            # Create synthesis prompt
            prompt = f"""You are a legal document analysis expert. Based on the retrieved information below, provide a comprehensive and well-structured answer to the user's query.

User Query: {query}

Retrieved Information:
{context}

Instructions:
1. If this is a comparison query (like "compare across all contracts"), organize your response to clearly show differences and similarities across documents
2. Provide specific citations with document names and pages
3. If the query asks for information across "all contracts" but only some documents contain relevant info, explicitly note this gap
4. Structure your response with clear headings and bullet points when appropriate
5. Be comprehensive but concise
6. If information appears incomplete, suggest what might be missing

Provide a thorough analysis that directly answers the user's question:"""
            
            # Get LLM response
            response = self.synthesis_llm.invoke(prompt)
            
            # Extract content from response
            if hasattr(response, 'content'):
                return response.content.strip()
            else:
                return str(response).strip()
                
        except Exception as e:
            logger.error(f"Error in LLM synthesis: {e}")
            return self.create_enhanced_answer(query, results)
    
    def prepare_llm_context(self, results: List[Dict]) -> str:
        """Prepare context for LLM synthesis"""
        context = ""
        for i, result in enumerate(results[:8], 1):  # Limit to top 8 for context length
            context += f"[Source {i}] Document: {result['document']}, Page: {result['page']}, Confidence: {result['confidence']:.3f}\n"
            content = result.get('full_content', result.get('text_excerpt', ''))[:400]  # Limit content length
            context += f"Content: {content}...\n\n"
        
        return context
    
    def deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        """Remove duplicate results and sort by confidence"""
        seen = set()
        unique_results = []
        
        for result in results:
            # Create unique key based on document and chunk
            key = f"{result['document']}_{result['chunk']}"
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        # Sort by confidence score
        return sorted(unique_results, key=lambda x: x['confidence'], reverse=True)
    
    def format_intent_based_response(self, query: str, results: List[Dict], analysis: Dict) -> str:
        """Format response based on query intent"""
        if not results:
            return "No relevant information found for your query."
        
        if analysis.get("comparison_intent", False):
            return self.format_comparison_response(results)
        elif analysis.get("cross_document", False):
            return self.format_cross_document_response(results)
        else:
            return self.create_enhanced_answer(query, results)
    
    def format_comparison_response(self, results: List[Dict]) -> str:
        """Format response for comparison queries"""
        # Group by document
        by_document = {}
        for result in results:
            doc = result["document"]
            if doc not in by_document:
                by_document[doc] = []
            by_document[doc].append(result)
        
        response = "## Document Comparison:\n\n"
        
        for doc, doc_results in by_document.items():
            response += f"**{doc}:**\n"
            for result in doc_results[:2]:  # Top 2 per document
                excerpt = result.get('text_excerpt', result.get('full_content', ''))[:200]
                response += f"- {excerpt}...\n"
            response += "\n"
        
        response += f"\n*Found relevant information in {len(by_document)} documents*"
        return response
    
    def format_cross_document_response(self, results: List[Dict]) -> str:
        """Format response for cross-document queries"""
        unique_docs = set(r["document"] for r in results)
        
        response = f"## Cross-Document Analysis (Found in {len(unique_docs)} documents):\n\n"
        
        for result in results[:5]:  # Top 5 results
            excerpt = result.get('text_excerpt', result.get('full_content', ''))[:300]
            response += f"**{result['document']}** (Page {result['page']}, {result['confidence']:.1%} confidence):\n"
            response += f"{excerpt}...\n\n"
        
        return response
    
    def create_enhanced_answer(self, query: str, results: List[Dict]) -> str:
        """Create enhanced answer without LLM"""
        if not results:
            return "No relevant information found for your query."
        
        answer = "Based on the relevant documents:\n\n"
        
        for result in results[:3]:  # Top 3 results
            if result["confidence"] > 0.5:
                content = result.get('full_content', result.get('text_excerpt', ''))
                answer += f"{content[:300]}...\n\n---\n\n"
        
        return answer.rstrip("\n---\n\n")


# Initialize enhanced service
enhanced_rag_service = EnhancedRAGService(
    groq_api_key=os.getenv("GROQ_API_KEY", "demo-key")
)