
from langchain.chains import LLMChain, SimpleSequentialChain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain, SequentialChain

def build_qa_chain(retriever, llm):
    
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False
    )

def build_summary_chain(llm):
    prompt = PromptTemplate(
        input_variables=["input"],
        template="You are a helpful AI assistant. Please improve and simplify this answer:\n\n{input}"
    )
    return LLMChain(llm=llm, prompt=prompt)

def build_simple_sequential_chain(qa_chain, summary_chain):
    return SimpleSequentialChain(chains=[qa_chain, summary_chain], verbose=True)

def build_conversational_retrieval_chain(retriever,llm, memory):
    return ConversationalRetrievalChain.from_llm( llm=llm, retriever=retriever, memory=memory)\
    
# def build_sequential_chain(conversational_chain, summary_chain):
#     return SequentialChain(
#         chains=[conversational_chain, summary_chain],
#         input_variables=["question", "chat_history"], 
#         output_variables=["output"], 
#         verbose=True
#     )    
