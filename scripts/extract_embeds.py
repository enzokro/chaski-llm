import argparse
import numpy as np
from pathlib import Path
from figma_llm.models.llm_manager import LLMManager
from figma_llm.utils.chunking import chunk_text

def parse_args():
    parser = argparse.ArgumentParser(description="Extract embeddings from input text")
    parser.add_argument("input_file", type=str, help="Path to the input text file")
    parser.add_argument("output_file", type=str, help="Path to the output file (numpy .npz)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the Llama model")
    parser.add_argument("--chunk_size", type=int, default=256, help="Chunk size for text splitting")
    parser.add_argument("--chunk_overlap", type=int, default=20, help="Chunk overlap for text splitting")
    return parser.parse_args()

def main():
    args = parse_args()

    # Load the input text
    with open(args.input_file, "r") as f:
        text = f.read()

    # Initialize the LLMManager
    llm_manager = LLMManager(args.model_path, embedding=True)

    # Chunk the text
    chunks = chunk_text(text, args.chunk_size, args.chunk_overlap)

    # Extract embeddings for each chunk
    embeddings = []
    for chunk in chunks:
        embedding = llm_manager.embed(chunk)
        embeddings.append(embedding)

    # Convert the list of embeddings to a numpy array
    embeddings_array = np.array(embeddings)

    # Save the embeddings to the output file
    output_file = Path(args.output_file)
    if output_file.suffix == ".npz":
        np.savez(output_file, embeddings=embeddings_array)
    else:
        raise ValueError("Output file must have a .npz extension")

    print(f"Embeddings extracted and saved to {args.output_file}")

if __name__ == "__main__":
    main()