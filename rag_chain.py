import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


PROMPT_TEMPLATE = """You are a helpful assistant. Use the context below to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."
Always mention the page number(s) where you found the answer.

Context:
{context}

Question: {question}

Answer:"""


def build_qa_chain(pdf_path: str):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(pages)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    prompt = PromptTemplate(
        template=PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )
    return chain, len(pages)


def ask(chain, question: str) -> dict:
    result = chain.invoke({"query": question})
    sources = sorted(
        {doc.metadata.get("page", 0) + 1 for doc in result["source_documents"]}
    )
    return {"answer": result["result"], "source_pages": sources}
