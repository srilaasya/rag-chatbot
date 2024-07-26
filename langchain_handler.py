import os
import dotenv
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough, RunnableBranch
from langchain_core.output_parsers import StrOutputParser

# Load environment variables
dotenv.load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# Initialize LangChain components


def initialize_langchain():
    global chat, retriever, document_chain, conversational_retrieval_chain

    os.environ["OPENAI_API_KEY"] = api_key
    chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

    # List of websites to scrape
    urls = [
        "https://www.artisan.co/",
        "https://www.artisan.co/pricing",
        "https://www.artisan.co/solutions/enterprise",
        "https://www.artisan.co/ai-sales-agent",
        "https://www.artisan.co/talk-to-sales?page_source=/",
        "https://www.artisan.co/sales-ai",
        "https://www.artisan.co/products/b2b-data",
        "https://www.artisan.co/products/sales-playbooks",
        "https://www.artisan.co/products/email-warmup",
        "https://www.artisan.co/products/sales-automation",
        "https://www.artisan.co/blog",
        "https://www.artisan.co/watch-demo",
        "https://www.artisan.co/casestudies",
        "https://www.artisan.co/solutions/midmarket",
        "https://www.artisan.co/about",
        "https://www.artisan.co/contact-us",
        "https://www.artisan.co/careers",
        "https://www.artisan.co/terms-of-use",
        "https://www.artisan.co/privacy-policy",
        "https://www.artisan.co/solutions/enterprise",
        "https://www.artisan.co/blog/how-to-find-b2b-leads",
        "https://www.artisan.co/blog/ai-sdr",
        "https://www.artisan.co/blog/b2b-sales-channels",
        "https://www.artisan.co/blog/lead-generation-outsourcing",
        "https://www.artisan.co/blog/apollo-alternatives",
        "https://www.artisan.co/blog/outbound-sales-software",
        "https://www.artisan.co/blog/ai-for-sales-prospecting",
        "https://www.artisan.co/blog/the-cold-emailing-masterclass-the-complete-guide-to-skyrocketing-leads-sales",
        "https://www.artisan.co/blog/artisan-raises-7-3-seed-round",
        "https://www.artisan.co/blog/outbound-sales-automation",
        "https://www.artisan.co/blog/ai-language-models-2024",
        "https://www.artisan.co/solutions/startups",
        "https://www.artisan.co/features/email-deliverability",
        "https://www.artisan.co/features/security"
    ]

    all_data = []
    for url in urls:
        loader = WebBaseLoader(url)
        data = loader.load()
        all_data.extend(data)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, chunk_overlap=0)
    all_splits = text_splitter.split_documents(all_data)

    vectorstore = Chroma.from_documents(
        documents=all_splits, embedding=OpenAIEmbeddings())
    retriever = vectorstore.as_retriever(k=4)

    SYSTEM_TEMPLATE = """
    Answer the user's questions based on the below context. 
    If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":
    
    <context>
    {context}
    </context>
    """

    question_answering_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_TEMPLATE),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    document_chain = create_stuff_documents_chain(
        chat, question_answering_prompt)

    query_transform_prompt = ChatPromptTemplate.from_messages(
        [
            MessagesPlaceholder(variable_name="messages"),
            (
                "user",
                "Given the above conversation, generate a search query to look up in order to get information relevant to the conversation. Only respond with the query, nothing else.",
            ),
        ]
    )

    query_transformation_chain = query_transform_prompt | chat

    query_transforming_retriever_chain = RunnableBranch(
        (
            lambda x: len(x.get("messages", [])) == 1,
            (lambda x: x["messages"][-1].content) | retriever,
        ),
        query_transform_prompt | chat | StrOutputParser() | retriever,
    ).with_config(run_name="chat_retriever_chain")

    conversational_retrieval_chain = RunnablePassthrough.assign(
        context=query_transforming_retriever_chain,
    ).assign(
        answer=document_chain,
    )


def process_user_message(message_content, conversation_context):
    conversation_context.append(HumanMessage(content=message_content))

    # Limit the conversation context to the last 10 messages
    if len(conversation_context) > 10:
        conversation_context.pop(0)

    response = conversational_retrieval_chain.invoke(
        {
            "messages": conversation_context
        }
    )

    # Append the AI's response to the conversation context
    conversation_context.append(AIMessage(content=response["answer"]))

    return response["answer"], conversation_context
