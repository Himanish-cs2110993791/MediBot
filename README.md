# 🩺 MediBot

**MediBot** is an AI-powered medical assistant that uses Streamlit, LangChain, FAISS, and HuggingFace’s Mistral-7B LLM. It allows users to interact with a chatbot trained on custom medical PDFs, offering quick, accurate, and sourced medical responses through a stylish interface.

---

## ✨ Features

- 💬 Chat with an AI Medical Assistant  
- 🔍 Semantic Search over Medical PDFs  
- 🧠 Vector DB with FAISS & Embeddings  
- 🎨 Custom UI using CSS animations  
- 📚 Cites sources with page references  
- ⚙️ Streamlit-based Web App  

---

## 🧰 Tech Stack

| Layer        | Tool/Library                                      |
|--------------|---------------------------------------------------|
| LLM          | `mistralai/Mistral-7B-Instruct-v0.3` via HuggingFace |
| Embeddings   | `sentence-transformers/all-MiniLM-L6-v2`          |
| Vector DB    | FAISS (`faiss-cpu`)                               |
| Framework    | LangChain                                         |
| Frontend     | Streamlit + Custom CSS                            |
| PDF Loader   | PyPDFium2                                         |
| Environment  | Python 3.x with `.env` for secrets                |

---

## 🚀 Installation

### 🔧 Dependencies (Option 1: Manual)
```bash
pip install streamlit langchain faiss-cpu huggingface_hub python-dotenv sentence-transformers pypdfium2
```

### 📦 Dependencies (Option 2: Requirements File)
```bash
pip install -r requirements.txt
```

### 🔐 Environment Setup
Create a `.env` file in the root directory:
```env
HF_TOKEN=your_huggingface_token_here
```

---

## 🗂️ Project Structure

```
.
├── medibot3.py                # Main Streamlit UI
├── medibot3_style.css         # Custom CSS animations & design
├── connect_memory_with_llm.py # CLI: Query vector DB + LLM
├── create_memory_for_llm2.py  # Load PDFs, create vectorstore
├── .env                       # Your HF token (ignored in .gitignore)
├── data/                      # Folder for medical PDFs
└── vectorstore/               # FAISS DB (auto-generated)
```

---

## 🧠 How It Works

### 🗃️ `create_memory_for_llm2.py`
- Loads medical PDFs from `data/`
- Splits them into semantic chunks
- Embeds using MiniLM
- Saves vectors to FAISS DB

### 🧪 `connect_memory_with_llm.py`
- Loads FAISS vector DB
- Uses LangChain to retrieve relevant chunks
- Passes context to Mistral LLM and displays sources

### 💬 `medibot3.py`
- Launches Streamlit UI with animated design
- Accepts user queries in chat
- Returns medically accurate answers with references

---

## 🖼️ Preview


![Screenshot (338)](https://github.com/user-attachments/assets/63fa3b74-2c57-4987-9f6b-b0cad33721e5)


---


## 📌 To-Do / Future Enhancements

- ✅ Add chat memory  
- ✅ Deploy to Streamlit Cloud  
- 🔲 Add voice input/output  
- 🔲 PDF upload via interface  
- 🔲 User login/auth system  

---

## 📜 License

This project is licensed under the **MIT License**.

---

## 👤 Author

**Built by a Computer Science undergrad specializing in AI.**  
For questions, collaborations, or support — please open an issue or email: [himanish3791.be21@chitkara.edu.in]
