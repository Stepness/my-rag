# my-rag

A small script that prompts the user to ask a cat-based question.<br>
The script creates a vector db of all the sentences in the fact file, create a vector of the query written by the user and then compares it with the db via a cosine similarity measure and puts it in a memory knowledge base.<br>
A language model then returns the calculated result.