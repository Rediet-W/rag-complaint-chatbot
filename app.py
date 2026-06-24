from src.retriever import ComplaintRetriever
from src.generator import ComplaintGenerator

# 1. Initialize components
retriever = ComplaintRetriever()
generator = ComplaintGenerator()

# 2. Ask a question
question = "What are the common issues with credit cards?"
context, sources = retriever.get_context(question)
answer = generator.generate(context, question)

# 3. Output
print(f"Question: {question}")
print(f"Answer: {answer}")
print("\nSources retrieved:")
for s in sources:
    print(f"- ID: {s.metadata.get('id')} | Text: {s.page_content[:50]}...")