# RAG-Based Complaint Chatbot Project

## Task 1: Data Cleaning and Preprocessing

**Objective:** Transform raw, unstructured consumer complaint data into a clean, ready-to-analyze dataset.

### Accomplishments

* **Data Filtering:** Filtered the raw dataset to focus on four high-impact product categories: *Credit card, Personal loan, Savings account, and Money transfer*.
* **Boilerplate Removal:** Applied text preprocessing to remove standard introductory phrases ("i am writing to file a complaint"), normalizing the narrative content to improve retrieval quality.
* **Data Sanitization:** Handled missing values by dropping empty entries in the `Consumer complaint narrative` column and performed case normalization to ensure consistency.
* **Persisted Output:** Saved the processed data to `../data/filtered_complaints.csv` for downstream use.

---

## Task 2: Text Chunking, Embedding, and Vector Store Indexing

**Objective:** Convert cleaned narratives into a searchable semantic vector space.

### Accomplishments

* **Stratified Sampling:** Implemented a stratified sampling approach to select 15,000 complaints. This ensures that the distribution of products in our vector store accurately mirrors the global distribution of the original dataset, preventing retrieval bias.
* **Chunking Strategy:** Utilized LangChain’s `RecursiveCharacterTextSplitter` with a `chunk_size` of 500 characters and a `chunk_overlap` of 50. This balance allows for granular, actionable information retrieval while maintaining sufficient surrounding context for accurate semantic interpretation.
* **Embedding Model:** Selected `sentence-transformers/all-MiniLM-L6-v2`. This model was chosen for its high efficiency and optimization for sentence-level semantic similarity, which is essential for capturing the intent behind user queries.
* **Vector Store:** Constructed a persistent vector store using **ChromaDB**. Each chunk is stored with metadata (original `Complaint ID` and `Product` category), enabling the system to trace retrieved information back to the source complaint.

---

## Technical Stack

* **Language:** Python
* **Data Processing:** `pandas`
* **Orchestration:** `LangChain`
* **Embeddings:** `HuggingFaceEmbeddings`
* **Vector Database:** `ChromaDB`

---

## Setup Instructions

1. **Install dependencies:**
```bash
pip install pandas langchain langchain-huggingface langchain-chroma

```


2. **Environment Setup:** Ensure your `HF_TOKEN` is set in your environment if you are accessing Hugging Face Hub (though model files are cached locally after the first run).
3. **Run Pipeline:** Execute the notebooks/scripts in order: `Task_1_Cleaning.ipynb` followed by `Task_2_Indexing.ipynb`.

