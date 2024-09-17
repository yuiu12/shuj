from langchain_openai import ChatOpenAI
from langchain_community.graphs import Neo4jGraph
from langchain.chains import GraphCypherQAChain
from langchain.prompts import PromptTemplate

llm = ChatOpenAI(
    openai_api_key="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"
)
graph = Neo4jGraph(
    url = "neo4j://localhost:7687",
    username = "neo4j",
    password = "neo4j1212"
)
CYPHER_GENERATION_TEMPLATE = """
You are an expert Neo4j Developer translating user questions into Cypher to answer questions about movies and provide recommendations.
Convert the user's question based on the schema.

Schema: {schema}
Question: {question}
"""
cypher_generation_prompt = PromptTemplate(
    template=CYPHER_GENERATION_TEMPLATE,
    input_variables=["schema","question"]
) 
cypher_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    cypher_prompt=cypher_generation_prompt,
    verbose=True
)
cypher_chain.invoke({"query":"What is the plot of the movie Toy Story?"})