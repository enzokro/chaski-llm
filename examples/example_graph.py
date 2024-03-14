from fastcore.xtras import Path
from chaski.graphs.builder import GraphBuilder
from chaski.utils.path_utils import get_outputs_dir, get_project_root

def main():
    """
    Builds a Knowledge Graph from the extracted page content from the Figma course:
    """
    # Set up the Inkbot graph builder, using its default prompts
    graph_builder = GraphBuilder()

    # Where to save the extracted knowledge graphs
    out_dir = get_outputs_dir() / 'graphs'
    out_dir.mkdir(exist_ok=True)

    # Point to the directory with the extracted Figma course content
    documents_dir = get_project_root() / "data/figma_documents"

    # NOTE: Add document formats to be read in here.
    doc_fmts = [
        ".txt",
    ]

    # Iterate over the documents in the directory
    for fid in documents_dir.ls().filter(lambda o: o.suffix in doc_fmts):

        # Read the document
        with open(fid, "r") as file:
            user_context = file.read()

        # Generate the Knowledge Graph
        knowledge_graph = graph_builder.build_graph(
            user_context=user_context,
            user_message="", # will use the default Graph-building user prompt.
        )

        # Print the generated Knowledge Graph
        print(f"Knowledge Graph for {fid.stem}:")
        print(knowledge_graph)
        print("-" * 50)

        # Save the Knowledge Graph to a file
        with open(out_dir / f"{fid.stem}_graph.txt", "w") as file:
            file.write(knowledge_graph)


if __name__ == "__main__":
    main()