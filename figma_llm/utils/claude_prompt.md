Thank you! However, the format of the chat template needs to be changed. It does not quite match the workflow for Retrieval-Augmented Generation (RAG). For more context, the workflow of RAG is given below:  

1. The user makes an initial query.  
2. Assuming there is an existing set of embeddings, we find the top-n most relevant (by cosine distance) embeddings. 
3. Then, we insert the text of these embeddings into the prompt, underneath the user's query. We also make it clear to the LLM that this is useful context, coming from a known knowledge base, to help guide its output generation. 
4. Then we end the prompt, including any final assistant/response setups as needed.  


With this in mind, please create a complete and working example, using all of our code so far, where we do the following: 

- Initialize the LLM Manager.   
- Initialize an empty embedding storage.  
- Embed a series of sample documents. 
- Save the embeddings to a file.   
- Load the embeddings from this saved file.  
- Now, use a new query, or input, that we will pass to the model.  
- Search the embeddings for the ones that are most similar to this query.  
- Add the matching text from the relevant embeddings as part of the prompt. 
- - Make sure the prompt is told this is "context" for its generation.
- Finish the generation task, using RAG via the context-augmented prompt with the text from the most similar embeddings. 
- Please add comments and explanations as needed in the script above. Follow the coding style of lucidrains and Jeremy Howard

The files are attached once again to refresh your context: `llm_manager.py`, `db.py` Please focus on completing the tasks above. 