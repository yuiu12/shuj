from langchain_openai import OpenAI

llm = OpenAI(openai_api_key="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")

response = llm.invoke("What is Neo4j?")

print(response)
from langchain.prompts import PromptTemplate

template = PromptTemplate(template="""
You are a cockney fruit and vegetable seller.
Your role is to assist your customer with their fruit and vegetable needs.
Respond using cockney rhyming slang.

Tell me about the following fruit: {fruit}
""", input_variables=["fruit"])

response = llm.invoke(template.format(fruit="apple"))

print(response)
