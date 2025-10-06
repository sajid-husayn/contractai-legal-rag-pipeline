"""
LLM Synthesis Service using Groq API for generating natural language answers from RAG results
"""
import os
import logging
from typing import List, Dict, Any
from groq import Groq

logger = logging.getLogger(__name__)

class GrokSynthesisService:
    def __init__(self):
        self.api_key = os.getenv("GROQ_API_KEY", "demo-key")
        
        # Available Groq FREE models (2025)
        self.available_models = {
            "openai/gpt-oss-20b": "OpenAI GPT-OSS 20B - Current default (3.6B active params)",
            "openai/gpt-oss-120b": "OpenAI GPT-OSS 120B - Flagship model (5.1B active params)",
            "llama3-70b-8192": "Llama 3 70B (8K context)",
            "llama3-8b-8192": "Llama 3 8B (8K context)",
            "llama-3.1-8b-instant": "Llama 3.1 8B Instant",
            "gemma2-9b-it": "Gemma 2 9B Instruct"
        }
        
        # Current model - change via GROQ_MODEL environment variable
        self.model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
        
        if not self.api_key or self.api_key == "demo-key":
            logger.warning("GROQ_API_KEY not found. LLM synthesis will be disabled.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)
            model_desc = self.available_models.get(self.model, "Unknown model")
            logger.info(f"Initialized Groq client with model: {self.model} ({model_desc})")
    
    def synthesize_answer(self, question: str, search_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Generate a natural language answer from RAG search results using Grok
        """
        if not self.client:
            return {
                "answer": "LLM synthesis unavailable - GROQ_API_KEY not configured",
                "model": "N/A"
            }
        
        try:
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results[:5]):  # Top 5 results
                document = result.get("document", "Unknown")
                text = result.get("full_content", result.get("text_excerpt", ""))
                confidence = result.get("confidence", 0)
                section = result.get("section_category", "general")
                
                context_parts.append(f"""
Source {i+1} ({document}, {section}, confidence: {confidence:.1%}):
{text[:500]}...
""")
            
            context = "\n".join(context_parts)
            
            # Create prompt for Grok
            prompt = f"""You are a legal document analysis assistant. Based on the following search results from legal documents, provide a clear, professional answer to the user's question.

Question: {question}

Search Results:
{context}

Instructions:
- Provide a direct, comprehensive answer to the question
- Cite specific documents and sections when relevant
- Be precise about legal terms and conditions
- If the information is incomplete, acknowledge what's missing
- Use professional legal language but keep it accessible
- Focus on answering the specific question asked

Answer:"""

            model_name = "openai/gpt-oss-20b"
            response = self.client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_completion_tokens=800,
                temperature=0.3,
                reasoning_effort="medium"
            )
            
            return {
                "answer": response.choices[0].message.content.strip(),
                "model": model_name
            }
            
        except Exception as e:
            logger.error(f"Error in LLM synthesis: {e}")
            return {
                "answer": f"Error generating synthesized answer: {str(e)}",
                "model": "openai/gpt-oss-20b"
            }

# Global instance
grok_synthesis_service = GrokSynthesisService()