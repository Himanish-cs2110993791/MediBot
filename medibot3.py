import streamlit as st
import os
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEndpoint
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
import pathlib

# Load environment variables
load_dotenv(find_dotenv())

# Constants
DB_FAISS_PATH = "vectorstore/db_faiss"
HUGGING_FACE_REPO_ID = "mistralai/Mistral-7B-Instruct-v0.3"

# Custom CSS for better UI
with open('medibot3_style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def ensure_directory_exists(path):
    """Ensure the directory exists, create if it doesn't"""
    pathlib.Path(path).parent.mkdir(parents=True, exist_ok=True)

@st.cache_resource
def get_vectorstore():
    try:
        ensure_directory_exists(DB_FAISS_PATH)
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        if not os.path.exists(DB_FAISS_PATH):
            st.error("Vector store directory not found. Please ensure the FAISS database is properly set up.")
            return None
            
        db = FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)
        return db
    except Exception as e:
        st.error(f"Error loading vector store: {str(e)}")
        return None

def set_custom_prompt(custom_prompt_template):
    try:
        prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context","question"])
        return prompt
    except Exception as e:
        st.error(f"Error setting up prompt template: {str(e)}")
        return None

def load_llm(huggingface_repo_id, HF_TOKEN):
    if not HF_TOKEN:
        st.error("HuggingFace token not found. Please set the HF_TOKEN environment variable.")
        return None
        
    try:
        llm = HuggingFaceEndpoint(
            repo_id=huggingface_repo_id,
            temperature=0.5,
            model_kwargs={"token": HF_TOKEN,
                          "max_length": 512}
        )
        return llm
    except Exception as e:
        st.error(f"Error loading LLM: {str(e)}")
        return None

def landing_page():
    st.markdown('<div class="landing-container">', unsafe_allow_html=True)
    
    # Hero Section with background animation
    st.markdown('<div class="hero-section">', unsafe_allow_html=True)
    st.markdown('<div class="hero-bg"></div>', unsafe_allow_html=True)
    st.markdown('<h1 class="hero-title">MediBot</h1>', unsafe_allow_html=True)
    st.markdown('<p class="hero-subtitle">Your AI-powered medical assistant, providing accurate and reliable medical information</p>', unsafe_allow_html=True)
    
    # Hero Image
    st.markdown('<img src="https://cdn-icons-png.flaticon.com/512/2966/2966321.png" class="hero-image" alt="MediBot">', unsafe_allow_html=True)
    
    # Capabilities Section
    st.markdown('<div class="capabilities">', unsafe_allow_html=True)
    
    capabilities = [
        {
            "title": "Medical Information",
            "description": "Access comprehensive medical knowledge from trusted sources. Get accurate information about diseases, treatments, and medical procedures.",
            "icon": "https://cdn-icons-png.flaticon.com/512/2966/2966321.png"
        },
        {
            "title": "Source Verification",
            "description": "Every response comes with verified sources and page references, ensuring the information is reliable and traceable.",
            "icon": "https://cdn-icons-png.flaticon.com/512/1570/1570887.png"
        },
        {
            "title": "Instant Responses",
            "description": "Get quick and accurate answers to your medical questions, helping you make informed decisions about your health.",
            "icon": "https://cdn-icons-png.flaticon.com/512/1570/1570887.png"
        },
        {
            "title": "Professional Knowledge",
            "description": "Powered by medical textbooks and professional resources, providing you with expert-level medical information.",
            "icon": "https://cdn-icons-png.flaticon.com/512/1570/1570887.png"
        }
    ]
    
    for capability in capabilities:
        st.markdown(f'''
            <div class="capability-card">
                <img src="{capability['icon']}" class="capability-icon" alt="{capability['title']}">
                <div class="capability-title">{capability['title']}</div>
                <div class="capability-description">{capability['description']}</div>
            </div>
        ''', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Start Chat Button
    if st.button("Start Chatting", key="start_chat", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

def chat_page():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Chat Header
    st.markdown('<div class="chat-header">', unsafe_allow_html=True)
    st.markdown('<h2 style="font-weight: 600; margin-bottom: 0.5rem;">MediBot Chat</h2>', unsafe_allow_html=True)
    st.markdown('<p style="color: #a0a0a0;">Ask me anything about medical information. I\'m here to help!</p>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Initialize chat history
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    st.markdown('<div class="main">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"], unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Chat input
    st.markdown('<div class="chat-input">', unsafe_allow_html=True)
    prompt = st.chat_input("Ask me anything about medical information...")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if prompt:
        # Add user message to chat history
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        CUSTOM_PROMPT_TEMPLATE = """
            Use the pieces of information provided in the context to answer user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.
            Do not include any references section in your answer.
            Provide a direct, informative, and medically accurate answer.

           
            Context: {context}
            Question: {question}

            Start the answer directly. No small talk please.
            """
        
        HF_TOKEN = os.environ.get("HF_TOKEN")
        
        try:
            # Show thinking animation
            thinking_container = st.empty()
            thinking_container.markdown("""
                <div class="thinking-animation">
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                    <div class="thinking-dot"></div>
                </div>
            """, unsafe_allow_html=True)
            
            vectorstore = get_vectorstore()
            if vectorstore is None:
                st.error("Failed to load Vector Store. Please check the setup.")
                return

            llm = load_llm(huggingface_repo_id=HUGGING_FACE_REPO_ID, HF_TOKEN=HF_TOKEN)
            if llm is None:
                st.error("Failed to load LLM. Please check your HuggingFace token.")
                return

            prompt_template = set_custom_prompt(CUSTOM_PROMPT_TEMPLATE)
            if prompt_template is None:
                st.error("Failed to set up prompt template.")
                return

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template} 
            )

            response = qa_chain.invoke({"query": prompt})
            result = response["result"]
            source_documents = response["source_documents"]

            # Clear thinking animation
            thinking_container.empty()

            # Extract book name and page number
            sources_info = "\n".join(
                f'<div class="source-item">{doc.metadata.get("source", "Unknown Book")}, Page {doc.metadata.get("page", "Unknown")}</div>'
                for doc in source_documents
            )
            # Check if the answer contains bullet points or numbered points
            result = result.replace("--------------------------------------------------", "").strip()
            result_formatted=result
            if any(marker in result for marker in ['•', '1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.', '-']):
            # Replace bullet points or numbered points with newline for better formatting
                result_formatted = result.replace('•', '\n•').replace('-', '\n-').replace('1.', '\n1.').replace('2.', '\n2.') \
                             .replace('3.', '\n3.').replace('4.', '\n4.').replace('5.', '\n5.').replace('6.', '\n6.') \
                                .replace('7.', '\n7.').replace('8.', '\n8.').replace('9.', '\n9.').replace('10.', '\n10.')
            # Clean the result to remove any unwanted dashed lines or extra formatting
            result_cleaned = result.replace("--------------------------------------------------", "").strip()

            # Clean extra bullet points and other unwanted characters
            result_cleaned = result_cleaned.replace("\n•", "\n").replace("\n-", "\n").replace("\n1.", "\n").replace("\n2.", "\n")

            # Prepare final output with styled source information
            result_to_show = f"""
<div style="line-height: 1.6; margin-bottom: 1rem;">
{result_formatted}
</div>

<div class="source-info">
    <div class="source-title">Sources:</div>
    {sources_info}
</div>
"""

            # Display in chatbot
            st.chat_message("assistant").markdown(result_to_show, unsafe_allow_html=True)
            st.session_state.messages.append({"role": "assistant", "content": result_to_show})

        except Exception as e:
            st.error(f"An error occurred while processing your query: {str(e)}")
            st.error("Please try again or contact support if the issue persists.")
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Initialize session state
    if 'page' not in st.session_state:
        st.session_state.page = "landing"
    
    # Show appropriate page based on session state
    if st.session_state.page == "landing":
        landing_page()
    else:
        chat_page()
            
if __name__ == "__main__":
    main()