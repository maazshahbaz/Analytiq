🌟 Analytiq – Unlock Insights from Institutional Data

**Analytiq** is an AI-powered platform designed for Institutional Research (IR) teams to streamline analysis of both unstructured and structured data. With Analytiq, you can:
- Upload PDFs, Word docs, spreadsheets (CSV, Excel), and more  
- Convert documents into a searchable vector store  
- Run natural language queries against your data  
- Explore SQL databases with a conversational agent  
- Perform in-depth DataFrame analysis via a pandas-powered agent  
- View, download, or export insights in CSV, Excel, or PDF  

---

## 🚀 Key Features
- **Document Agent**: Chat with your uploaded documents; retrieves and cites source files  
- **SQL Explorer**: Connect to SQLite (or other databases) and ask questions via SQL or plain English  
- **Pandas Agent**: Upload CSV/Excel and run advanced analyses with memory support  
- **Modular Architecture**: Separate `agents/`, `config/`, `scripts/`, and `utils/` folders for scalability  
- **FastAPI Backend**: REST endpoints for Docs, SQL, and Pandas

---

## 🛠️ Getting Started

### Prerequisites
- Python 3.8 or later  
- Git  

### Installation
```bash
# Clone the repository
git clone https://github.com/maazshahbaz/Analytiq.git
cd Analytiq

# Set up a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root with:
```
OPENAI_API_KEY=<your_openai_key>
COHERE_API_KEY=<your_cohere_key>  # if using reranking
```

### Run the App
```bash
uvicorn app:app --reload
```

Then open http://localhost:8000/docs to explore the API.
We include smoke-test scripts under `scripts/`:
- `run_database_agent.py`: Verifies the SQL agent can list tables and run queries
- `run_unstructured_agent.py`: Tests the HybridQAChain against sample queries
- `run_pandas_agent.py`: Tests DataFrame agent with an in-memory dummy DataFrame
- `eval_unstructured.py`: RAG evaluation harness using gold.jsonl

Run any script via:
```bash
python -m scripts/run_database_agent
```
(Replace with the appropriate script name.)

## 📁 Project Structure
```
Analytiq/
├── agents/
│   ├── database_agent/      # SQL agent code & UI
│   ├── pandas_agent/        # DataFrame agent code & UI
│   └── unstructured_agent/  # Document loader, vector store, chain, UI
├── config/                  # Global settings & .env loader
├── data/                    # Persistent storage (vector DB, sample DB)
├── scripts/                 # Smoke-test and evaluation scripts
├── utils/                   # Shared helpers (session storage, prompts)
├── app.py                  # FastAPI entrypoint
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

## 🤝 Contributing
Contributions are welcome! Please fork the repo, create a feature branch, and submit a pull request.
For large changes, open an issue first to discuss.

## 📄 License
This project is licensed under the MIT License.

## 👤 Author
maazshahbaz
