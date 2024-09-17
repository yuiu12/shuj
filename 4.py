from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser

chat_llm = ChatOpenAI(openai_api_key="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")

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a surfer dude, having a conversation about the surf conditions on the beach. Respond using surfer slang.",
        ),
        (
            "human", 
            "{question}"
        ),
    ]
)

chat_chain = prompt | chat_llm | StrOutputParser()

response = chat_chain.invoke({"question": "What is the weather like?"})

print(response)