import streamlit as st
import os
import pickle
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Optional
import hashlib

# PDF processing
import pdfplumber

# Vector database
import faiss

# OpenAI integration
from openai import OpenAI

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

# Configuration
EMBEDDING_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
VECTOR_STORE_PATH = "vector_store.pkl"
DOCUMENTS_PATH = "documents.pkl"

class DocumentProcessor:
    """Handles PDF processing and text extraction"""
    
    @staticmethod
    def extract_text_from_pdf(pdf_file) -> str:
        """Extract text from uploaded PDF file"""
        try:
            text = ""
            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
            return text
        except Exception as e:
            raise Exception(f"Error extracting text from PDF: {str(e)}")
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        """Split text into overlapping chunks"""
        if not text:
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < len(text):
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > start + chunk_size // 2:
                    chunk = text[start:break_point + 1]
                    end = break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
            
            if start >= len(text):
                break
                
        return [chunk for chunk in chunks if chunk]

class VectorStore:
    """Handles vector embeddings and FAISS operations"""
    
    def __init__(self):
        self.index = None
        self.texts = []
        self.metadata = []
        self.dimension = 1536  # text-embedding-3-small dimension
        
    def create_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Create embeddings using OpenAI"""
        try:
            embeddings = []
            batch_size = 100  # Process in batches to avoid rate limits
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                response = openai_client.embeddings.create(
                    model=EMBEDDING_MODEL,
                    input=batch
                )
                batch_embeddings = [item.embedding for item in response.data]
                embeddings.extend(batch_embeddings)
                
            return embeddings
        except Exception as e:
            raise Exception(f"Error creating embeddings: {str(e)}")
    
    def add_documents(self, texts: List[str], metadata: List[Dict]):
        """Add documents to vector store"""
        if not texts:
            return
            
        # Create embeddings
        embeddings = self.create_embeddings(texts)
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatL2(self.dimension)
            
        # Convert embeddings to numpy array
        embedding_array = np.array(embeddings, dtype=np.float32)
        
        # Add to FAISS index
        self.index.add(embedding_array)
        
        # Store texts and metadata
        self.texts.extend(texts)
        self.metadata.extend(metadata)
    
    def search(self, query: str, k: int = 5) -> List[Dict]:
        """Search for similar documents"""
        if self.index is None or len(self.texts) == 0:
            return []
            
        try:
            # Create query embedding
            query_embedding = self.create_embeddings([query])[0]
            query_vector = np.array([query_embedding], dtype=np.float32)
            
            # Search in FAISS
            distances, indices = self.index.search(query_vector, k)
            
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(self.texts):
                    results.append({
                        'text': self.texts[idx],
                        'metadata': self.metadata[idx],
                        'score': float(distance),
                        'rank': i + 1
                    })
            
            return results
        except Exception as e:
            raise Exception(f"Error searching documents: {str(e)}")
    
    def save(self, filepath: str):
        """Save vector store to file"""
        try:
            data = {
                'index': faiss.serialize_index(self.index) if self.index else None,
                'texts': self.texts,
                'metadata': self.metadata,
                'dimension': self.dimension
            }
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            st.error(f"Error saving vector store: {str(e)}")
    
    def load(self, filepath: str):
        """Load vector store from file"""
        try:
            if os.path.exists(filepath):
                with open(filepath, 'rb') as f:
                    data = pickle.load(f)
                
                if data['index']:
                    self.index = faiss.deserialize_index(data['index'])
                self.texts = data.get('texts', [])
                self.metadata = data.get('metadata', [])
                self.dimension = data.get('dimension', 1536)
                
                return True
        except Exception as e:
            st.error(f"Error loading vector store: {str(e)}")
        return False

class RAGSystem:
    """Main RAG system coordinator"""
    
    def __init__(self):
        self.vector_store = VectorStore()
        self.documents = {}
        
    def add_document(self, filename: str, content: str):
        """Add a document to the RAG system"""
        try:
            # Create document hash for deduplication
            doc_hash = hashlib.md5(content.encode()).hexdigest()
            
            if doc_hash in self.documents:
                return False, "Document already exists"
            
            # Process document
            chunks = DocumentProcessor.chunk_text(content)
            
            if not chunks:
                return False, "No text could be extracted from document"
            
            # Create metadata for each chunk
            metadata = []
            for i, chunk in enumerate(chunks):
                metadata.append({
                    'filename': filename,
                    'chunk_id': i,
                    'doc_hash': doc_hash,
                    'upload_time': datetime.now().isoformat()
                })
            
            # Add to vector store
            self.vector_store.add_documents(chunks, metadata)
            
            # Store document info
            self.documents[doc_hash] = {
                'filename': filename,
                'content': content,
                'chunk_count': len(chunks),
                'upload_time': datetime.now().isoformat()
            }
            
            return True, f"Successfully processed {len(chunks)} chunks"
            
        except Exception as e:
            return False, f"Error processing document: {str(e)}"
    
    def search_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Search for relevant document chunks"""
        return self.vector_store.search(query, k)
    
    def generate_answer(self, query: str, context: List[Dict]) -> str:
        """Generate answer using retrieved context"""
        try:
            # Prepare context from retrieved documents
            context_text = "\n\n".join([
                f"Document: {item['metadata']['filename']}\n{item['text']}"
                for item in context
            ])
            
            # Create prompt
            system_prompt = """You are a helpful assistant that answers questions based on the provided context. 
            Use only the information from the context to answer questions. If the context doesn't contain 
            enough information to answer the question, say so clearly. Be concise but comprehensive."""
            
            user_prompt = f"""Context:
{context_text}

Question: {query}

Please provide a detailed answer based only on the context above."""

            # Generate response
            response = openai_client.chat.completions.create(
                model=CHAT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating answer: {str(e)}"
    
    def save_data(self):
        """Save all data to files"""
        self.vector_store.save(VECTOR_STORE_PATH)
        try:
            with open(DOCUMENTS_PATH, 'wb') as f:
                pickle.dump(self.documents, f)
        except Exception as e:
            st.error(f"Error saving documents: {str(e)}")
    
    def load_data(self):
        """Load data from files"""
        self.vector_store.load(VECTOR_STORE_PATH)
        try:
            if os.path.exists(DOCUMENTS_PATH):
                with open(DOCUMENTS_PATH, 'rb') as f:
                    self.documents = pickle.load(f)
        except Exception as e:
            st.error(f"Error loading documents: {str(e)}")

def init_session_state():
    """Initialize Streamlit session state"""
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = RAGSystem()
        st.session_state.rag_system.load_data()
    
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    if 'processing' not in st.session_state:
        st.session_state.processing = False

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="RAG System",
        page_icon="📚",
        layout="wide"
    )
    
    # Initialize session state
    init_session_state()
    
    st.title("📚 RAG System - Document Q&A")
    st.markdown("Upload PDF documents and ask questions about their content using AI-powered retrieval and generation.")
    
    # Check OpenAI API key
    if not openai_client.api_key:
        st.error("⚠️ OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        st.stop()
    
    # Sidebar for document management
    with st.sidebar:
        st.header("📄 Document Management")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload PDF Document",
            type=['pdf'],
            help="Upload a PDF file to add to the knowledge base"
        )
        
        if uploaded_file and not st.session_state.processing:
            if st.button("Process Document", type="primary"):
                st.session_state.processing = True
                st.rerun()
        
        # Process uploaded file
        if st.session_state.processing and uploaded_file:
            with st.spinner("Processing document..."):
                try:
                    # Extract text from PDF
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                        tmp_file.write(uploaded_file.read())
                        tmp_file.flush()
                        
                        text = DocumentProcessor.extract_text_from_pdf(tmp_file.name)
                        os.unlink(tmp_file.name)
                    
                    # Add to RAG system
                    success, message = st.session_state.rag_system.add_document(
                        uploaded_file.name, text
                    )
                    
                    if success:
                        st.success(f"✅ {message}")
                        st.session_state.rag_system.save_data()
                    else:
                        st.error(f"❌ {message}")
                        
                except Exception as e:
                    st.error(f"❌ Error processing file: {str(e)}")
                
                finally:
                    st.session_state.processing = False
                    st.rerun()
        
        # Display document statistics
        st.subheader("📊 Knowledge Base")
        doc_count = len(st.session_state.rag_system.documents)
        chunk_count = len(st.session_state.rag_system.vector_store.texts)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Documents", doc_count)
        with col2:
            st.metric("Text Chunks", chunk_count)
        
        # List documents
        if st.session_state.rag_system.documents:
            st.subheader("📋 Documents")
            for doc_hash, doc_info in st.session_state.rag_system.documents.items():
                with st.expander(doc_info['filename']):
                    st.write(f"**Chunks:** {doc_info['chunk_count']}")
                    st.write(f"**Uploaded:** {doc_info['upload_time'][:19]}")
                    if st.button(f"Delete", key=f"del_{doc_hash}"):
                        # Note: In a full implementation, you'd need to remove 
                        # the chunks from the vector store as well
                        del st.session_state.rag_system.documents[doc_hash]
                        st.session_state.rag_system.save_data()
                        st.rerun()
    
    # Main chat interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for i, (role, content, timestamp) in enumerate(st.session_state.chat_history):
                if role == "user":
                    st.chat_message("user").write(f"**{timestamp}**\n\n{content}")
                else:
                    st.chat_message("assistant").write(f"**{timestamp}**\n\n{content}")
        
        # Chat input
        if chunk_count == 0:
            st.info("📝 Please upload some PDF documents first to start asking questions.")
        else:
            query = st.chat_input("Ask a question about your documents...")
            
            if query:
                # Add user message to history
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append(("user", query, timestamp))
                
                # Search for relevant context
                with st.spinner("Searching documents..."):
                    context = st.session_state.rag_system.search_documents(query, k=5)
                
                # Generate answer
                with st.spinner("Generating answer..."):
                    answer = st.session_state.rag_system.generate_answer(query, context)
                
                # Add assistant response to history
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.chat_history.append(("assistant", answer, timestamp))
                
                st.rerun()
    
    with col2:
        st.header("🔍 Context Sources")
        
        # Show sources for the last query if available
        if st.session_state.chat_history:
            last_user_message = None
            for role, content, _ in reversed(st.session_state.chat_history):
                if role == "user":
                    last_user_message = content
                    break
            
            if last_user_message:
                try:
                    context = st.session_state.rag_system.search_documents(last_user_message, k=3)
                    
                    if context:
                        st.subheader("📖 Retrieved Sources")
                        for i, item in enumerate(context, 1):
                            with st.expander(f"Source {i}: {item['metadata']['filename']}"):
                                st.write(f"**Relevance Score:** {item['score']:.3f}")
                                st.write(f"**Chunk:** {item['metadata']['chunk_id'] + 1}")
                                st.write("**Content:**")
                                st.write(item['text'][:300] + "..." if len(item['text']) > 300 else item['text'])
                    else:
                        st.info("No relevant sources found.")
                except:
                    pass
        
        # Chat history management
        st.subheader("📋 Chat History")
        chat_count = len([msg for msg in st.session_state.chat_history if msg[0] == "user"])
        st.metric("Messages", chat_count * 2)  # User + Assistant messages
        
        if st.button("Clear Chat History", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()

if __name__ == "__main__":
    main()
