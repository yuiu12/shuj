from langchain_community.graphs import Neo4jGraph
graph = Neo4jGraph(
    url = "neo4j://localhost:7687",
    username = "neo4j",
    password = "neo4j1212"
)
result = graph.query("""
MATCH (m:Movie{title: 'Toy Story'}) 
RETURN m.title, m.plot, m.poster
""")

print(result)