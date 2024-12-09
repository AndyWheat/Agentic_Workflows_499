from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List
import operator
import json
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage, ChatMessage
import nest_asyncio
nest_asyncio.apply()

import re
from pathlib import Path
import os

from langchain_openai import ChatOpenAI
from llama_index.llms.openai import OpenAI
api_key = "insert api key"

model = None
llm = None

def initialize_models(api_key):
    global model,llm
    #model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    #llm = OpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
    model = ChatOpenAI(model="gpt-3.5-turbo")
    llm = OpenAI(model="gpt-3.5-turbo")
    return model, llm



from pydantic import BaseModel
import os
import json
from fuzzywuzzy import fuzz
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, SummaryIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.tools import FunctionTool, QueryEngineTool
from llama_index.core.vector_stores import MetadataFilters, FilterCondition
from typing import List, Optional
import PyPDF2
from llama_index.core import VectorStoreIndex
from llama_index.core.objects import ObjectIndex
from llama_index.core.agent import FunctionCallingAgentWorker, AgentRunner

import streamlit as st
import json
from langgraph.checkpoint.sqlite import SqliteSaver

class Queries(BaseModel):
    queries: List[str]


# main.py

from RAG_toolset import *

# #IMPORTANT CODE THAT RUNS THINGS 
# #-------------------------------------
# #-------------------------------------

# # Define the root folder where the PDFs and metadata are stored
# documents_folder = Path("C:/Users/Andrew/Teacher_Documents/Textbooks/")

# # Gather all the PDFs in the folder and its subfolders
# papers = [str(paper) for paper in documents_folder.rglob("*.pdf")]

# # Dictionary to store the tools for each paper
# paper_to_tools_dict = {}

# # Loop through each PDF and get the tools
# for paper in papers:
#     # Extract the chapter title from the PDF file path
#     chapter_title = extract_chapter_title(paper)
    
#     # Ensure that the tool name is within the allowed length
#     print(f"Processing: {chapter_title}")
#     short_name = generate_tool_name(chapter_title)
    
#     # Get the corresponding metadata file
#     metadata_file = get_metadata_file(paper)
    
#     if metadata_file:
#         # Call the get_doc_tools function with the PDF file and metadata
#         try:
#             vector_tool, summary_tool = get_doc_tools(paper, metadata_file, short_name)
#             paper_to_tools_dict[paper] = [vector_tool, summary_tool]
#             print(f"Tools created for {paper}")
#         except Exception as e:
#             print(f"Error processing {paper}: {str(e)}")
#     else:
#         print(f"No metadata found for {paper}, skipping...")


# # Output the resulting dictionary of tools
# print(f"Tools generated for {len(paper_to_tools_dict)} papers.")


# # Define tools and index as you have
# all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]

# # Create object index and retriever
# obj_index = ObjectIndex.from_objects(
#     all_tools,
#     index_cls=VectorStoreIndex,
# )

# obj_retriever = obj_index.as_retriever(similarity_top_k=3)

# # Initialize a list to store the query-response history
# interaction_history = []

# # Custom function to add the history of interactions
# def log_interaction(query, response):
#     # Access metadata if it exists, or provide an empty dictionary
#     metadata = getattr(response, 'metadata', {})
#     interaction_history.append({
#         "query": query,
#         "response": response.content if hasattr(response, 'content') else str(response),
#         "metadata": metadata
#     })

# # Customize the system prompt to emphasize logging metadata
# system_prompt = """ \
# You are an agent designed to answer queries over a set of given papers.
# For each query, you will answer based on the tools provided. Log each question, response, and related metadata.
# Always provide the filename from the metadata in each response.
# """


# # Define the agent worker with logging
# agent_worker = FunctionCallingAgentWorker.from_tools(
#     tool_retriever=obj_retriever,
#     llm=llm, 
#     system_prompt=system_prompt,
#     verbose=False
# )

# # Create an agent runner with logging support
# class LoggingAgentRunner(AgentRunner):
#     def query(self, question):
#         # Call the base query method to get the response
#         response = super().query(question)
#         # Log the interaction
#         log_interaction(question, response)
#         return response

# agent = LoggingAgentRunner(agent_worker)


# obj_index = ObjectIndex.from_objects(
#         all_tools,
#         index_cls=VectorStoreIndex,
#     )

# #----------------------------------------------
# #----------------------------------------------


# EXAM_WRITER_PROMPT = """You are a course director tasked with creating an exam based on the material given to you.\
# Make sure that all the material you add is directly from the textbook.\
# Write an exam with 10-15 questions.\
# You MUST write an even mix of short answer, multiple choice, and fill in the blank.\
# Assign point values for each question, make the questions with multiple concepts worth more.\

# Utilize all the information below:

# --------

# {textbook_info}
# """

# WRITER_ASSISTANT_PROMPT = """You are a teaching assistant tasked with researching the textbook for the exam writer.\
# Generate a list of questions based on the topic to search about in the textbook to help write an exam. 
# """

# MOCK_STUDENT_PROMPT = """You are a knowledgeable student answering an exam.
# Answer each question with the correct response based on your understanding of the material provided in the textbook.
# Provide clear and accurate answers, whether the question is multiple choice, fill-in-the-blank, or short answer.
# For each question, refer to the concepts, definitions, or formulas as they appear in the textbook to simulate correct responses.

# Here is the exam to answer:

# {exam}
# """


# GRADER_PROMPT = """You are an exam grader tasked with grading an exam taken by the user.\
# Provide a grade out of 100 based on the user input compared to the answer sheet.\
# Provide feedback for each individual response and cite where the material can be found based on the answer sheet.\

# This is the answer sheet: {answer_sheet}\
# These are the student's responses: {user_input}\

# ### Format your response as follows:\
# Total Score: [numerical score]/100\

# Question Scores:\
# 1. [Score for question 1]/[Point value] | Feedback for question 1\
# 2. [Score for question 2]/[Point value] | Feedback for question 2\
# ... (repeat for each question)\
# """

# REVISION_PROMPT = """You are a student designed to ask for additional help on whatever you missed.\
# Take any feedback about the questions you missed and ask for an exam that covers those topics.\
# Here is the feedback : {feedback}
# """

# from typing import List, TypedDict
# import re

# class AgentState(TypedDict):
#     topic: str
#     exam: str
#     answer_sheet: str
#     user_input: str
#     textbook_info: str
#     current_score: int
#     feedback: str
#     content: str
        

# # Initialize a list to store the query-response history
# interaction_history = []

# # Custom function to add the history of interactions
# def log_interaction(query, response):
#     # Access metadata if it exists, or provide an empty dictionary
#     metadata = getattr(response, 'metadata', {})
#     interaction_history.append({
#         "query": query,
#         "response": response.content if hasattr(response, 'content') else str(response),
#         "metadata": metadata
#     })



# def writer_assistant_node(state: AgentState):
#     # Prepare the list of queries based on the state topic
#     queries = model.with_structured_output(Queries).invoke([
#         SystemMessage(content=WRITER_ASSISTANT_PROMPT),
#         HumanMessage(content=state['topic'])
#     ])
    
#     # Clear interaction history at the start to avoid accumulation across multiple calls
#     interaction_history.clear()
    
#     # Query the agent and log each interaction for every question
#     for q in queries['queries']:
#         response = agent.query(q)
#         log_interaction(q, response)  # Use the logging function for each query
    
#     # Format the interaction history as a single string for `textbook_info`
#     textbook_info = "\n".join(
#         f"Query: {entry['query']}\nResponse: {entry['response']}\nMetadata: {entry['metadata']}"
#         for entry in interaction_history
#     )
    
#     return {"textbook_info": textbook_info}


# def writer_node(state: AgentState):
#     textbook_info = str(state.get('textbook_info', ''))

#     messages = [
#         SystemMessage(content=EXAM_WRITER_PROMPT), 
#         HumanMessage(content=textbook_info)
#     ]
#     response = model.invoke(messages)
#     #print("User Input: " +state["user_input"])
#     return {
#         "exam": response.content.strip()  # Only the exam questions
#     }

# def mock_student_node(state: AgentState):
#     # Use textbook information in the mock student prompt to ensure answers are based on it
#     messages = [
#         SystemMessage(content=MOCK_STUDENT_PROMPT),
#         HumanMessage(content=f"{state['exam']}\n\nRefer to the following textbook information:\n\n{state['textbook_info']}")
#     ]
#     response = model.invoke(messages)
#     st.write("Here is the exam: " + state["exam"])
#     return {
#         "answer_sheet": response.content.strip()  # The model answer sheet based on textbook info
#     }


# def grader_node(state: AgentState):
#     messages = [
#         SystemMessage(content=GRADER_PROMPT.format(answer_sheet=state['answer_sheet'], user_input=state['user_input']))
#     ]
#     response = model.invoke(messages)

#     # Debug: Print the response to verify the format
#     st.write("Grader response content:", response.content)
    
#     # Regex to capture the total score out of 100
#     total_score_match = re.search(r"Total Score: (\d+)/100", response.content)
#     total_score = int(total_score_match.group(1)) if total_score_match else 0
    
#     # Extract individual question scores and feedback
#     question_scores = []
#     question_score_pattern = re.compile(r"(\d+)\.\s+(\d+)/(\d+)\s*\|\s*(.*)")
#     for match in question_score_pattern.finditer(response.content):
#         question_number = int(match.group(1))  # Question number
#         score = int(match.group(2))            # Score for the question
#         max_points = int(match.group(3))       # Max points for the question
#         feedback = match.group(4).strip()      # Feedback for the question
        
#         question_scores.append({
#             "question_number": question_number,
#             "score": score,
#             "max_points": max_points,
#             "feedback": feedback
#         })
    
#     # Return total score, individual question scores, and feedback
#     return {
#         "current_score": total_score,
#         "question_scores": question_scores
#     }

# def revision_node(state: AgentState):
#     messages = [
#         SystemMessage(content = REVISION_PROMPT.format(feedback = state['feedback']))
#     ]
#     response = model.invoke(messages)
#     return {"feedback": response.content}


# def should_continue(state: AgentState):
#     st.write("Current score in should_continue:", state["current_score"])  # Debug line
#     return END if state["current_score"] > 90 else "revision"


# # Define the `student_input_node` function
# def student_input_node(state: AgentState):
#     # Placeholder function where user will provide answers during runtime
#     pass



# # Initialize StateGraph and add nodes
# builder = StateGraph(AgentState)
# builder.add_node("writer_assistant", writer_assistant_node)
# builder.add_node("writer", writer_node)
# builder.add_node("mock_student", mock_student_node)  # New node for mock answer sheet
# builder.add_node("student_input", student_input_node)
# builder.add_node("grader", grader_node)
# builder.add_node("revision", revision_node)

# # Define entry point and edges
# builder.set_entry_point("writer_assistant")
# builder.add_edge("writer_assistant", "writer")
# builder.add_edge("writer", "mock_student")  # Move to mock student to create answer sheet
# builder.add_edge("mock_student", "student_input")  # Move to student input after answer sheet is ready
# builder.add_edge("student_input", "grader")  # Grade after student input

# # Conditional edge for grading - continue if score < 90, else end
# builder.add_conditional_edges("grader", should_continue, {END: END, "revision": "revision"})
# builder.add_edge("revision", "writer_assistant")  # Restart the loop for revision


# import json
# from langgraph.checkpoint.sqlite import SqliteSaver

# def json_safe_print(data):
#     """Recursively convert data to JSON-serializable format where possible."""
#     if isinstance(data, dict):
#         return {k: json_safe_print(v) for k, v in data.items()}
#     elif isinstance(data, list):
#         return [json_safe_print(item) for item in data]
#     elif isinstance(data, (str, int, float, bool, type(None))):
#         return data  # JSON serializable types
#     else:
#         return str(data)  # Convert non-serializable objects to strings
# 

from langgraph.checkpoint.memory import MemorySaver


graph = None
thread = None
obj_index = None
agent = None
exam = None
feedback = None
interaction_history = []


def initialize_graph():
    """Set up the graph, tools, nodes, and necessary objects."""
    global graph, thread, obj_index, agent, interaction_history, upload_folder  # Declare globals

    if graph and thread:  # Prevent reinitialization
        return {"message": "Graph and tools already initialized"}

    print("Initializing graph and processing documents...")
    print("made it here")
    try:
        # Define the root folder where the PDFs and metadata are stored
        print(upload_folder)
        upload_folder = upload_folder.strip('"')  # Removes extra quotes
        upload_folder = upload_folder.replace("\\", "/")
        print(upload_folder)
        documents_folder = Path(upload_folder)
        print(documents_folder)
        # Gather all the PDFs in the folder and its subfolders
        if not documents_folder.exists():
            print(f"Error: The folder '{documents_folder}' does not exist.")
            raise FileNotFoundError(f"The folder '{documents_folder}' does not exist.")
        else:
            print(f"The folder '{documents_folder}' exists.")

        # Gather all the PDFs in the folder and its subfolders
        papers = list(documents_folder.rglob("*.pdf"))
        if not papers:
            print(f"No PDF files found in '{documents_folder}'.")
        else:
            print(f"Found {len(papers)} PDF files: {papers}")


        # Dictionary to store the tools for each paper
        paper_to_tools_dict = {}

        # Loop through each PDF and get the tools
        for paper in papers:
            #chapter_title = extract_chapter_title(paper)  # Extract chapter title
            print(f"Processing: {paper}")
            short_name = generate_tool_name(str(paper))  # Generate tool name
            print(short_name)
            #metadata_file = get_metadata_file(paper)  # Retrieve corresponding metadata

            try:
                # Create tools for the document
                vector_tool, summary_tool = get_doc_tools(paper, short_name)
                paper_to_tools_dict[paper] = [vector_tool, summary_tool]
                print(f"Tools created for {paper}")
            except Exception as e:
                print(f"Error processing {paper}: {str(e)}")

        print(f"Tools generated for {len(paper_to_tools_dict)} papers.")

        # Define tools and indices
        all_tools = [t for paper in papers for t in paper_to_tools_dict[paper]]
        obj_index = ObjectIndex.from_objects(all_tools, index_cls=VectorStoreIndex)
        obj_retriever = obj_index.as_retriever(similarity_top_k=3)

        # Initialize a list to store the query-response history
        interaction_history = []

        # Function to log interactions
        def log_interaction(query, response):
            metadata = getattr(response, 'metadata', {})
            interaction_history.append({
                "query": query,
                "response": response.content if hasattr(response, 'content') else str(response),
                "metadata": metadata
            })

        # Define agent and worker
        system_prompt = """ \
        You are an agent designed to answer queries over a set of given papers.
        For each query, you will answer based on the tools provided. Log each question, response, and related metadata.
        Answer as in depth as you possibly can and check as much of the material as necessary.
        Always provide the filename from the metadata in each response.
        """

        agent_worker = FunctionCallingAgentWorker.from_tools(
            tool_retriever=obj_retriever,
            llm=llm,
            system_prompt=system_prompt,
            verbose=False
        )

        class LoggingAgentRunner(AgentRunner):
            def query(self, question):
                response = super().query(question)
                log_interaction(question, response)
                return response

        agent = LoggingAgentRunner(agent_worker)


        # Prompts
       
        EXAM_WRITER_PROMPT = """You are a course director tasked with creating an exam based on the material provided below. 
        Your task is to generate an exam formatted as JSON. Follow these guidelines:
        1. The exam must include 10-15 questions, with an even mix of:
        - Short answer
        - Multiple choice
        - Fill in the blank
        2. Assign point values for each question:
        - Questions involving multiple concepts should have higher point values.
        3. Include clear options for multiple-choice questions.
        4. The JSON structure MUST follow this format:
        {
            "questions": [
                {
                    "number": <question_number>,
                    "type": <"short_answer" | "multiple_choice" | "fill_in_the_blank">,
                    "text": <question_text>,
                    "options": [<options_for_multiple_choice_if_applicable>],
                    "answer": <correct_answer>,
                    "points": <point_value>
                },
                ...
            ]
        }

        Utilize the material below to ensure all questions are directly derived from it:

        --------

        {textbook_info}
        """

        WRITER_ASSISTANT_PROMPT = """You are a teaching assistant tasked with researching the textbook for the exam writer.\
        Generate a list of questions based on the topic to search about in the textbook to help write an exam. 
        """

        MOCK_STUDENT_PROMPT = """You are a knowledgeable student answering an exam.
        Answer each question with the correct response based on your understanding of the material provided in the textbook.
        Provide clear and accurate answers, whether the question is multiple choice, fill-in-the-blank, or short answer.
        For each question, refer to the concepts, definitions, or formulas as they appear in the textbook to simulate correct responses.

        Here is the exam to answer:

        {exam}
        """


        GRADER_PROMPT = """You are an exam grader tasked with grading an exam taken by the user.\
        Provide a grade out of 100 based on the user input compared to the answer sheet.\
        Provide feedback for each individual response and cite where the material can be found based on the answer sheet.\

        This is the answer sheet: {answer_sheet}\
        These are the student's responses: {user_input}\

        ### Format your response as follows:\
        Total Score: [numerical score]/100\

        Question Scores:\
        1. [Score for question 1]/[Point value] | Feedback for question 1\
        2. [Score for question 2]/[Point value] | Feedback for question 2\
        ... (repeat for each question)\
        """

        REVISION_PROMPT = """You are a student designed to ask for additional help on whatever you missed.\
        Take any feedback about the questions you missed and ask for an exam that covers those topics.\
        Here is the feedback : {feedback}
        """


        class AgentState(TypedDict):
            topic: str
            exam: str
            answer_sheet: str
            user_input: str
            textbook_info: str
            current_score: int
            feedback: str
            content: str
            
        # Define node functions
        def writer_assistant_node(state: AgentState):
            # Prepare the list of queries based on the state topic
            queries = model.with_structured_output(Queries).invoke([
                SystemMessage(content=WRITER_ASSISTANT_PROMPT),
                HumanMessage(content=state['topic'])
            ])
            
            # Clear interaction history at the start to avoid accumulation across multiple calls
            interaction_history.clear()
            
            # Query the agent and log each interaction for every question
            for q in queries['queries']:
                response = agent.query(q)
                log_interaction(q, response)  # Use the logging function for each query
            
            # Format the interaction history as a single string for textbook_info
            textbook_info = "\n".join(
                f"Query: {entry['query']}\nResponse: {entry['response']}\nMetadata: {entry['metadata']}"
                for entry in interaction_history
            )
            print(textbook_info)
            return {"textbook_info": textbook_info}


        def writer_node(state: AgentState):
            textbook_info = str(state.get('textbook_info', ''))

            messages = [
                SystemMessage(content=EXAM_WRITER_PROMPT), 
                HumanMessage(content=textbook_info)
            ]
            response = model.invoke(messages)
            #print("User Input: " +state["user_input"])
            return {
                "exam": response.content.strip()  # Only the exam questions
            }

        def mock_student_node(state: AgentState):
            global exam
            # Use textbook information in the mock student prompt to ensure answers are based on it
            messages = [
                SystemMessage(content=MOCK_STUDENT_PROMPT),
                HumanMessage(content=f"{state['exam']}\n\nRefer to the following textbook information:\n\n{state['textbook_info']}")
            ]
            response = model.invoke(messages)
            print("Here is the exam: " + state["exam"])
            exam = state["exam"]
            return {
                "answer_sheet": response.content.strip()  # The model answer sheet based on textbook info
            }


        def grader_node(state: AgentState):
            global feedback
            messages = [
                SystemMessage(content=GRADER_PROMPT.format(answer_sheet=state['answer_sheet'], user_input=state['user_input']))
            ]
            response = model.invoke(messages)
            
            # Regex to capture the total score out of 100
            total_score_match = re.search(r"Total Score: (\d+)/100", response.content)
            total_score = int(total_score_match.group(1)) if total_score_match else 0
            # Extract individual question scores and feedback
            question_scores = []
            question_score_pattern = re.compile(r"(\d+)\.\s+(\d+)/(\d+)\s*\|\s*(.*)")
            for match in question_score_pattern.finditer(response.content):
                question_number = int(match.group(1))  # Question numberTool
                score = int(match.group(2))            # Score for the question
                max_points = int(match.group(3))       # Max points for the question
                feedback = match.group(4).strip()      # Feedback for the question
                
                question_scores.append({
                    "question_number": question_number,
                    "score": score,
                    "max_points": max_points,
                    "feedback": feedback
                })
            print("Made it here in grader!!\n")
            feedback = question_scores
            # Return total score, individual question scores, and feedback
            print(question_scores)
            return {
                "current_score": total_score,
                "question_scores": question_scores
            }

        def revision_node(state: AgentState):
            messages = [
                SystemMessage(content = REVISION_PROMPT.format(feedback = state['feedback']))
            ]
            response = model.invoke(messages)
            return {"feedback": response.content}


        def should_continue(state: AgentState):
            print("Current score in should_continue:", state["current_score"])  # Debug line
            return END if state["current_score"] > 90 else "revision"


        # Define the student_input_node function
        def student_input_node(state: AgentState):
            # Placeholder function where user will provide answers during runtime
            pass

        # Initialize StateGraph
        builder = StateGraph(AgentState)
        builder.add_node("writer_assistant", writer_assistant_node)
        builder.add_node("writer", writer_node)
        builder.add_node("mock_student", mock_student_node)
        builder.add_node("student_input", student_input_node)
        builder.add_node("grader", grader_node)
        builder.add_node("revision", revision_node)

        builder.set_entry_point("writer_assistant")

        builder.add_edge("writer_assistant", "writer")
        builder.add_edge("writer", "mock_student")
        builder.add_edge("mock_student", "student_input")
        builder.add_edge("student_input", "grader")
        builder.add_conditional_edges("grader", should_continue, {END: END, "revision": "revision"})
        builder.add_edge("revision", "writer_assistant")

        memory = MemorySaver()
        graph = builder.compile(checkpointer=memory, interrupt_before=["student_input"])

        thread = {"configurable": {"thread_id": "1"}}
        graph.update_state(thread, {
            "topic": "",
            "exam": "",
            "answer_sheet": "",
            "user_input": "",
            "textbook_info": "",
            "current_score": 0,
            "feedback": "",
            "content": ""
        })

        print("Graph and tools initialized successfully.")
        return graph, thread

    except Exception as e:
        print(f"Error during initialization: {e}")
        return {"error": str(e)}



#Flask Code

from flask import Flask, render_template, request, jsonify
from langgraph.checkpoint.sqlite import SqliteSaver

app = Flask(__name__)

# Initialize global variables for the graph and thread
graph = None
thread = None
interaction_history = []
upload_folder = "insert path here"


@app.route('/select_folder', methods=['POST'])
def select_folder():
    global upload_folder
    data = request.json
    if not data or 'folderPath' not in data:
        return jsonify({"error": "No folder path provided"}), 400

    upload_folder = data['folderPath']  # Save the folder path as a string
    return jsonify({"message": "Folder path saved successfully", "folderPath": upload_folder}), 200



@app.route('/set_api_key', methods=['POST'])
def set_api_key():
    global api_key
    data = request.json
    if not data or 'api_key' not in data:
        return jsonify({"error": "API key is required"}), 400
    api_key = data['api_key']
    return jsonify({"message": "API key updated successfully"}), 200



@app.route('/')
def index():
    """Serve the main HTML page."""
    return render_template('index.html')


@app.route('/initialize', methods=['POST'])
def initialize():
    """Initialize the graph and set up the initial state."""
    global api_key
    print("made it here 1")
    initialize_models(api_key)
    try:
        # Call the `initialize_graph` function to set up everything
        result = initialize_graph()

        if "error" in result:
            return jsonify(result), 500  # If initialization fails, return an error response

        # No need to set `graph` and `thread` again because they're already global
        return jsonify({"message": "Graph initialized successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def json_safe_print(data):
    """Recursively convert data to JSON-serializable format where possible."""
    if isinstance(data, dict):
        return {k: json_safe_print(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [json_safe_print(item) for item in data]
    elif isinstance(data, (str, int, float, bool, type(None))):
        return data  # JSON serializable types
    else:
        return str(data)  # Convert non-serializable objects to strings


import re


@app.route('/execute', methods=['POST'])
def execute_graph():
    """Execute the graph with a dynamically updated topic and return the JSON-formatted exam."""
    global graph, thread

    if not graph or not thread:
        return jsonify({"error": "Graph has not been initialized"}), 400

    try:
        # Get the topic dynamically from the POST request
        topic = request.json.get("topic", "").strip()

        if not topic:
            return jsonify({"error": "Topic cannot be empty"}), 400

        # Update the graph state with the new topic
        graph.update_state(thread, {"topic": topic})

        # Stream the graph execution to completion
        for s in graph.stream(graph.get_state(thread).values, thread):
            pass

        # Retrieve the updated state from the graph
        state = graph.get_state(thread).values
        exam_json = state.get("exam", "")  # The "exam" field should already contain JSON content

        # Print the raw exam content for debugging
        print(f"Raw exam content: {exam_json}")

        # Parse the exam JSON if necessary
        try:
            # If the exam is already a dict, no need to parse
            if isinstance(exam_json, str):
                exam_data = json.loads(exam_json)  # Parse the JSON string into a Python dictionary
            else:
                exam_data = exam_json
        except json.JSONDecodeError as e:
            return jsonify({"error": f"Failed to parse exam JSON: {str(e)}"}), 500

        # Return the parsed JSON directly
        return jsonify({"exam": exam_data}), 200
    except Exception as e:
        return jsonify({"error": f"Graph execution failed: {str(e)}"}), 500


@app.route('/submit_answers', methods=['POST'])
def submit_answers():
    """Handle answer submissions."""
    global graph, thread, feedback

    if not graph or not thread:
        return jsonify({"error": "Graph has not been initialized"}), 400

    # Collect answers from the request
    try:
        answers = request.json  # JSON data from the request
        print("Received answers:", answers)
        graph.update_state(thread, {"user_input": answers}, as_node="student_input")
        for event in graph.stream(None, thread, stream_mode="values"):
            #print(json.dumps(json_safe_print(event), indent=4))
            pass #This a false print because I need the for loop
        # Process answers or store them as needed
        # For example, updating the graph with user input or grading logic
        print(feedback)
        return jsonify({"message": "Answers received successfully", "feedback": feedback}), 200
    except Exception as e:
        return jsonify({"error": f"Failed to process answers: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)







#Streamlit Code
"""
def main():
    st.title("Exam Simulation with LangGraph and Streamlit")
    
    # Initialize the graph with SqliteSaver as the checkpointer
    with SqliteSaver.from_conn_string(":memory:") as checkpointer:
        graph = builder.compile(checkpointer=checkpointer, interrupt_before=["student_input"])

        initial_state = {
            "topic": "Give me an exam on Probability Rules",
            "exam": "",
            "answer_sheet": "",
            "user_input": "",
            "textbook_info": "",
            "current_score": 0,
            "feedback": "",
            "content": ""
        }
        
        thread = {"configurable": {"thread_id": "1"}}
        graph.update_state(thread, initial_state)

        # Run the graph execution loop
        iteration = 0  # Track iteration count to use in unique keys
        while True:
            # Display node execution up to the student_input node
            for s in graph.stream(graph.get_state(thread).values, thread):
                st.json(json_safe_print(s.values))  # Display state in Streamlit as JSON

            # Check if the current score is satisfactory
            current_state = graph.get_state(thread).values
            if current_state["current_score"] > 90:
                st.write("Score is above 90. Ending the loop.")
                break  # End loop if score requirement is met

            # Check if we are waiting for user input at student_input
            if not current_state["user_input"]:
                user_input = st.text_input(
                    "Please enter your answers for the exam:",
                    key=f"user_input_{iteration}"  # Unique key for each iteration
                )
                if st.button("Submit Answer", key=f"submit_button_{iteration}"):  # Unique key for each iteration
                    st.write("Proceeding to grading...")
                    
                    # Update graph state with user input
                    graph.update_state(thread, {"user_input": user_input}, as_node="student_input")
                    
                    # Restart the loop to process the user input
                    st.experimental_rerun()
                else:
                    st.stop()  # Pause execution until user interacts

            # Display node execution from grader after student_input
            for event in graph.stream(None, thread, stream_mode="values"):
                st.json(json_safe_print(event.values))  # Display state in Streamlit as JSON

            iteration += 1  # Increment iteration for unique key in the next loop

if __name__ == "__main__":
    main()
"""