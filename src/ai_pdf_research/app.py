from langchain_google_genai import ChatGoogleGenerativeAI
import streamlit as st
import os
from pydantic import SecretStr
from ingest import load_and_split_pdf, store_chunks_in_chroma, get_retriever
from chains import build_qa_chain, build_summary_chain, build_simple_sequential_chain,build_conversational_retrieval_chain
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory
from langchain.memory.chat_message_histories import StreamlitChatMessageHistory


def main():
    load_dotenv()
    uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")
    UPLOAD_DIR = "uploads"
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    if uploaded_file is not None:
        # Save file
        with open(f"{UPLOAD_DIR}/my_file.pdf", "wb") as f:
            f.write(uploaded_file.getbuffer())
            st.success("File saved successfully!")
            chunks = load_and_split_pdf(f"{UPLOAD_DIR}/my_file.pdf")
            # Set a default ChromaDB directory if not specified in environment
            chroma_db = os.getenv("CHROMA_DB_DIR", "./chroma_db")
            store_chunks_in_chroma(chunks, persist_directory=chroma_db)
            st.session_state["ready_for_qa"] = True
    else:
        st.info("Please upload a PDF to continue.")

    if st.session_state.get("ready_for_qa"):
        mode = st.radio("Choose QA mode", ["Single-turn", "Conversational"])
        st.title("AI PDF Research Assistant")
        query = st.text_input("Ask a question about the PDF content:")
        if query:
            st.write(f"You asked: {query}")    
            if mode == "Single-turn":
                run_chains(query)
            else:
                run_conversational_chain(query)




def run_chains(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("MODEL")  
    
    if query:
        
        chroma_db_path = "./chroma_db"
        if os.path.exists(chroma_db_path):
            print(f"Vector database found at: {os.path.abspath(chroma_db_path)}")

            with st.spinner("Processing your question..."):
                try:
                    retriever = get_retriever(chroma_db_path)
                    if retriever is None:
                        st.error("Failed to create retriever from vector database.")
                        return

                    print(f"{model} model is being used")
                    lm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)

                    retrievalQaChain = build_qa_chain(retriever, llm=lm)

                    summary_chain = build_summary_chain(lm)
                    final_chain = build_simple_sequential_chain(retrievalQaChain, summary_chain)

                    final_answer = final_chain.run(query)
                    st.markdown(f"**Refined Answer:** {final_answer}")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    print(f"Detailed error: {e}")
        else:
            st.error("Vector database not found. Please upload a PDF first.")


def run_conversational_chain(query):
    api_key = os.getenv("GOOGLE_API_KEY")
    model = os.getenv("MODEL")  
    # if st.button("🧹 Clear Chat History"):
    #     st.session_state.chat_history = StreamlitChatMessageHistory().clear()
    #     st.success("Chat history cleared.")
    #     return

    
    if query:
        
        chroma_db_path = "./chroma_db"
        if os.path.exists(chroma_db_path):
            print(f"Vector database found at: {os.path.abspath(chroma_db_path)}")

            with st.spinner("Processing your question..."):
                try:
                    retriever = get_retriever(chroma_db_path)
                    if retriever is None:
                        st.error("Failed to create retriever from vector database.")
                        return

                    print(f"{model} model is being used")
                    lm = ChatGoogleGenerativeAI(model=model, google_api_key=api_key)
                    if "chat_history" not in st.session_state:
                        st.session_state.chat_history = StreamlitChatMessageHistory()
                    memory = ConversationBufferMemory(
                        chat_memory=st.session_state.chat_history,
                        return_messages=True,
                        memory_key="chat_history"
                    )


                    conversationalChain = build_conversational_retrieval_chain(retriever, lm, memory)

                    result = conversationalChain.invoke({"question": query})
                    answer = result["answer"]
                    # Output
                    st.markdown(f"**Conversation Answer:** {answer}")
                    chat_history = memory.load_memory_variables({}).get("chat_history", [])

                    if chat_history:
                        st.markdown("Conversation History")
                        for msg in chat_history:
                            role = "🧑 You" if msg.type == "human" else "🤖 AI"
                            st.markdown(f"**{role}:** {msg.content}")
                    else:
                        st.info("No chat history yet.")

                except Exception as e:
                    st.error(f"Error processing query: {str(e)}")
                    print(f"Detailed error: {e}")
        else:
            st.error("Vector database not found. Please upload a PDF first.")            


if __name__ == "__main__":
    main()
