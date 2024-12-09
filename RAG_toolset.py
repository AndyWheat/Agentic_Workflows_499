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
model = ChatOpenAI(model="gpt-3.5-turbo")
llm = OpenAI(model="gpt-3.5-turbo")

from pydantic import BaseModel

class Queries(BaseModel):
    queries: List[str]

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



# # Function to extract text from a PDF and split by pages

# def extract_pdf_text_by_page(pdf_file):
#     """Extracts text from a PDF file and splits it by pages."""
#     with open(pdf_file, "rb") as f:
#         reader = PyPDF2.PdfReader(f)
#         pages_text = [reader.pages[page].extract_text() for page in range(len(reader.pages))]
#     return pages_text

# # Function to match metadata paragraphs with PDF content using fuzzy matching
# def match_metadata_with_pdf(metadata_sections, pdf_page_texts):
#     """Match metadata sections with PDF content using fuzzy matching."""
#     matched_sections = []
#     pdf_text = "\n".join(pdf_page_texts)  # Combine all pages to fuzzy match across the entire document

#     for section in metadata_sections:
#         section_text = " ".join(section['lines'])  # Combine lines into a full section
#         # Use fuzzy matching to find the best match in the PDF text
#         best_match_page = max(pdf_page_texts, key=lambda page_text: fuzz.partial_ratio(section_text[:100], page_text))
#         matched_sections.append((section, best_match_page))
#     return matched_sections

# def check_node_metadata(nodes):
#     """Checks each node for metadata and prints the result."""
#     for idx, node in enumerate(nodes):
#         print(f"Checking metadata for node {idx + 1}:")
#         metadata = node.metadata
#         if metadata:
#             print(f"  - Filename: {metadata.get('filename', 'N/A')}")
#             print(f"  - Section Header: {metadata.get('section_header', 'N/A')}")
#             print(f"  - Section Text (excerpt): {metadata.get('section_text', 'N/A')[:100]}...")
#             print(f"  - PDF Section Start: {metadata.get('pdf_section_start_text', 'N/A')}")
#             print(f"  - Chapter Title: {metadata.get('chapter_title', 'N/A')}")
#         else:
#             print("  - No metadata found.")

# def get_doc_tools(file_path: str, metadata_path: str, name: str) -> str:
#     """Get vector query and summary query tools from a document, with metadata linking."""

#     # Step 1: Load the document (PDF)
#     documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
#     splitter = SentenceSplitter(chunk_size=1024)
#     nodes = splitter.get_nodes_from_documents(documents)

#     # Step 2: Load metadata (JSON) associated with the PDF
#     if os.path.exists(metadata_path):
#         with open(metadata_path, "r", encoding="utf-8") as f:
#             metadata = json.load(f)
#     else:
#         print(f"Metadata file {metadata_path} not found!")
#         metadata = {}

#     # Step 3: Extract text from the PDF and split by page
#     pdf_page_texts = extract_pdf_text_by_page(file_path)

#     # Step 4: Match metadata sections with PDF text
#     matched_sections = match_metadata_with_pdf(metadata.get("sections", []), pdf_page_texts)

#     # Step 5: Link the matched metadata to the corresponding text in the PDF
#     for node, (meta_section, pdf_section_start) in zip(nodes, matched_sections):
#         # Add filename, section header, and matched metadata to each node
#         node.metadata["filename"] = str(file_path)
#         node.metadata["section_header"] = meta_section.get("header", "N/A")
#         node.metadata["section_text"] = " ".join(meta_section.get("lines", []))
#         node.metadata["pdf_section_start_text"] = pdf_section_start  # Add the matched start of the PDF section
#         node.metadata["chapter_title"] = metadata.get("chapter_title", "Unknown Chapter")

#     # Step 5b: Check if metadata is correctly added to the nodes
#     #check_node_metadata(nodes)

#     # Step 6: Create a VectorStoreIndex with the nodes containing metadata
#     vector_index = VectorStoreIndex(nodes)
    
#     def vector_query(
#         query: str, 
#         page_numbers: Optional[List[str]] = None
#     ) -> str:
#         """Use to answer questions over a given paper, with optional filtering by page numbers."""

#         # **FIX HERE**: Appending all metadata to metadata_dicts
#         metadata_dicts = []
#         if page_numbers:
#             metadata_dicts.append({"key": "page_label", "value": page_numbers})
        
#         # Gather metadata for each node and append it to the metadata_dicts for filtering
#         for node in nodes:
#             for key, value in node.metadata.items():
#                 metadata_dicts.append({"key": key, "value": value})
        
#         # Query engine filters by metadata (such as page number or paragraph index)
#         query_engine = vector_index.as_query_engine(
#             similarity_top_k=2,
#             filters=MetadataFilters.from_dicts(metadata_dicts, condition=FilterCondition.OR)
#         )
#         response = query_engine.query(query)

#         # **FIX HERE**: Show metadata for each response node
#         #for node in response.source_nodes:
#             #print(f"Node text: {node.get_text()[:50]}...")  # Print an excerpt of the node text
#             #print(f"Node metadata: {node.metadata}")        # Show metadata

#         return response

#     # Ensure the tool name is valid (max 40 characters, only valid characters)
#     short_name = generate_tool_name(name)

#     # Step 7: Create a tool for vector-based queries
#     vector_query_tool = FunctionTool.from_defaults(
#         name=f"vector_tool_{short_name}",
#         fn=vector_query
#     )
    
#     summary_index = SummaryIndex(nodes)
#     summary_query_engine = summary_index.as_query_engine(
#         response_mode="tree_summarize",
#         use_async=True,
#     )
#     summary_tool = QueryEngineTool.from_defaults(
#         name=f"summary_tool_{name}",
#         query_engine=summary_query_engine,
#         description=(
#             f"Useful for summarization questions related to {name}"
#         ),
#     )

#     return vector_query_tool, summary_tool



# # Helper function to extract chapter title portion from the file path and remove "Computational Probability and Statistics"
# def extract_chapter_title(file_path: str) -> str:
#     """Extracts the 'Chapter XX Title' portion of the file name and removes 'Computational Probability and Statistics'."""
#     # Extract the base file name without the directory path and extension
#     file_name = os.path.basename(file_path).replace(".pdf", "")
    
#     # Remove the 'Computational Probability and Statistics' portion from the title
#     cleaned_name = re.sub(r'\s+Computational Probability and Statistics', '', file_name)
    
#     # Use regex to extract the "Chapter XX Title" portion (before any additional content)
#     match = re.search(r'(Chapter \d+ [\w\s-]+)', cleaned_name)
    
#     if match:
#         return match.group(1)
#     else:
#         # Fallback if the pattern doesn't match; return the cleaned name limited to 40 characters
#         return cleaned_name[:40]

# # Function to get the metadata file corresponding to the PDF
# def get_metadata_file(pdf_file):
#     """Returns the path to the metadata file corresponding to the PDF."""
#     metadata_file = pdf_file.replace(".pdf", "_metadata.json")
#     return metadata_file if Path(metadata_file).exists() else None

# Helper function to ensure function name is valid
def generate_tool_name(name: str) -> str:
    """Generates a valid tool name that only contains valid characters."""
    # Remove any invalid characters (anything that's not alphanumeric, _, or -)
    valid_name = re.sub(r'[^a-zA-Z0-9_-]', '', name)
    
    # Limit the name to the last 40 characters
    max_length = 40
    return valid_name[-max_length:]



def get_doc_tools(
    file_path: str,
    name: str,
) -> str:
    """Get vector query and summary query tools from a document."""

    # load documents
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    splitter = SentenceSplitter(chunk_size=1024)
    nodes = splitter.get_nodes_from_documents(documents)
    vector_index = VectorStoreIndex(nodes)
    
    def vector_query(
        query: str, 
        page_numbers: Optional[List[str]] = None
    ) -> str:
        """Use to answer questions over a given paper.
    
        Useful if you have specific questions over the paper.
        Always leave page_numbers as None UNLESS there is a specific page you want to search for.
    
        Args:
            query (str): the string query to be embedded.
            page_numbers (Optional[List[str]]): Filter by set of pages. Leave as NONE 
                if we want to perform a vector search
                over all pages. Otherwise, filter by the set of specified pages.
        
        """
    
        page_numbers = page_numbers or []
        metadata_dicts = [
            {"key": "page_label", "value": p} for p in page_numbers
        ]
        
        query_engine = vector_index.as_query_engine(
            similarity_top_k=2,
            filters=MetadataFilters.from_dicts(
                metadata_dicts,
                condition=FilterCondition.OR
            )
        )
        response = query_engine.query(query)
        return response
        
    
    vector_query_tool = FunctionTool.from_defaults(
        name=f"vector_tool_{name}",
        fn=vector_query
    )
    
    summary_index = SummaryIndex(nodes)
    summary_query_engine = summary_index.as_query_engine(
        response_mode="tree_summarize",
        use_async=True,
    )
    summary_tool = QueryEngineTool.from_defaults(
        name=f"summary_tool_{name}",
        query_engine=summary_query_engine,
        description=(
            f"Useful for summarization questions related to {name}"
        ),
    )

    return vector_query_tool, summary_tool
