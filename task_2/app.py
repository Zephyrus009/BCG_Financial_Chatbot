import streamlit as st
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.messages import AIMessage, HumanMessage
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Loading the LLM
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash",temperature=0.9)

#loading the vector db
vectorstore = Chroma(
    persist_directory=r"D:\BCG_Financial_Chatbot\task_2\bcg_fin_vectorstore", embedding_function=HuggingFaceEmbeddings()
)
retriever = vectorstore.as_retriever()

# Incorporate the retriever into a question-answering chain.
system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "\n\n"
    "{context}"
)

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# Incorporate the retriver in History retrival chain
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)

contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    llm, retriever, contextualize_q_prompt
)

## Running the Retrival Chain (RAG)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# Use the RAG Chain
def reposnse_generator(question, chat_history):

    ai_msg = rag_chain.invoke({"input": question,"chat_history": chat_history})

    chat_history.extend(
        [
            HumanMessage(content=question),
            AIMessage(content=ai_msg["answer"]),
        ]
    )

    print(chat_history)

    response = ai_msg["answer"]

    return response, chat_history



############################### Streamlit Chatbot UI ########################################################

PAGE_CONFIG = {"page_title":"BCG Finchat", 
               "page_icon":"./bcg_x_logo.png", 
               "layout":"centered", 
               "initial_sidebar_state":"auto"}

st.set_page_config(**PAGE_CONFIG)

st.image("./bcg_x_logo.png", caption="BCG X")
st.title("BCG Financial Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# initialize the chat history for RAG
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Write Something Here"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    response, chat_history = reposnse_generator(prompt,st.session_state.chat_history)

    # Update session state with new chat history
    st.session_state.chat_history = chat_history

    with st.chat_message("assistant"):
        st.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})