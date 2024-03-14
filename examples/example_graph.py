from fastcore.xtras import Path
from chaski.graphs.graph_builder import GraphBuilder

def main():
    """
    Builds a Knowledge Graph from the extracted page content from the Figma course:
    """
    # Set up the Inkbot graph builder, using its default prompts
    graph_builder = GraphBuilder()

    # Point to the directory with the extracted Figma course content
    documents_dir = Path("figma_llm/graphs/documents")

    # Add documents to read in here.
    doc_fmts = [
        ".txt",
    ]

    # Where to save the output knowledge graphs
    out_dir = Path("data/")

    # Iterate over the documents in the directory
    for fid in documents_dir.ls().filter(lambda o: o.suffix in doc_fmts):

        # Read the document content
        with open(fid, "r") as file:
            user_context = file.read()

        # Generate the Knowledge Graph using the GraphBuilder
        knowledge_graph = graph_builder.build_graph(
            user_context=user_context,
            user_message="", # will use the default Graph-building user prompt.
        )

        # Print the generated Knowledge Graph
        print(f"Knowledge Graph for {fid.stem}:")
        print(knowledge_graph)
        print("-" * 50)

        with open(out_dir / f"{fid.stem}_graph.txt", "w") as file:
            file.write(knowledge_graph)

if __name__ == "__main__":
    main()