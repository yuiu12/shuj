from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage  

chat_llm = ChatOpenAI(
    openai_api_key="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"
)
instructions = SystemMessage(content="""
You are a surfer dude, having a conversation about the surf conditions on the beach.
Respond using surfer slang.
""")
question = HumanMessage(content="What is the weather like?")
response = chat_llm.invoke([
    instructions,
    question
])

print(response.content)