from langchain_core.prompts import PromptTemplate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

class ComplaintGenerator:
    def __init__(self, model_id="google/flan-t5-base"):
        # 1. Load model and tokenizer directly
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)
        
        # 2. Define the prompt template
        self.template = """
        You are a financial analyst assistant. Answer based ONLY on the context.
        Context: {context}
        Question: {question}
        Answer:
        """
        self.prompt = PromptTemplate(template=self.template, input_variables=["context", "question"])

    def generate(self, context, question):
        if not context or not context.strip():
            return "No relevant context found to answer the question."
            
        # Format the prompt
        formatted_prompt = self.prompt.format(context=context, question=question)
        
        # Generate using the model directly
        inputs = self.tokenizer(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
        outputs = self.model.generate(**inputs, max_new_tokens=150)
        
        # Decode the output
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)