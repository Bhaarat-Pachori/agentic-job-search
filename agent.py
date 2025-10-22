from dotenv import load_dotenv
load_dotenv() 

import os
import time
from typing import List, TypedDict, Optional
from pypdf import PdfReader 
from langchain_google_genai import ChatGoogleGenerativeAI

# --- IMPORTS ---
from langchain_google_community.search import GoogleSearchAPIWrapper

# --- IMPORTS for Prompting and Output Parsing ---
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field # For structured output
import json # For handling potential JSON errors

# --- NEW: Import LangGraph components ---
from langgraph.graph import StateGraph, END

# print("\n--- DETAILED ENV VAR CHECK ---")
# print(f"LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING')}")
# print(f"LANGSMITH_ENDPOINT: {os.environ.get('LANGSMITH_ENDPOINT')}")
# api_key_check = os.environ.get('LANGSMITH_API_KEY')
# print(f"LANGSMITH_API_KEY: ...{api_key_check[-5:]}" if api_key_check else "Not Set") 
# print(f"LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT')}") # <-- Should print JobAgent
# print("-----------------------------\n")

# ==================================================================
# 1. DEFINE OUR AGENT'S "STATE"
# ==================================================================
# --- NEW: Define the structure we WANT the LLM to output ---
# This uses Pydantic for validation and description
class JobAnalysisOutput(BaseModel):
    job_title: str = Field(description="The job title identified from the listing.")
    is_match: str = Field(description="Rate the match as 'High', 'Medium', 'Low', or 'No Match' based ONLY on required skills and experience mentioned.")
    missing_skills: List[str] = Field(description="List of specific skills explicitly required by the job that are NOT found in the resume.")
    citizenship_required: Optional[str] = Field(description="Specify if US or Canadian citizenship is explicitly required (e.g., 'US Citizen Only', 'Canadian Citizen Only', or null).")
    project_suggestion: Optional[str] = Field(description="If missing_skills is not empty, suggest a brief 1-2 sentence project idea incorporating one or more missing skills. Otherwise, null.")
    posted_date: Optional[str] = Field(description="Date the job was posted, if mentioned (e.g., '2025-10-20', '3 days ago', or null).")


class JobAnalysis(TypedDict):
    job_title: str
    is_match: str
    missing_skills: List[str]
    citizenship_required: Optional[str]
    project_suggestion: Optional[str]
    posted_date: Optional[str]


class AgentState(TypedDict):
    resume_text: str
    search_queries: List[str] # <-- We will provide this at the start
    job_listings: List[dict]  # <-- This node will fill this key
    analysis_results: List[JobAnalysis]
    final_report: str

# ==================================================================
# 2. INITIALIZE LLM 
# ==================================================================
# --- Initialize the LLM globally ---
# We can use "gemini-1.5-flash" (faster, cheaper) or "gemini-1.5-pro" (more powerful)
# Low temp for consistent analysis
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)

# ==================================================================
# 3. DEFINE OUR "NODES"
# ==================================================================

def read_resume(state: AgentState):
    """Reads the resume_text from a PDF file."""
    print("--- Reading Resume ---")
    resume_path = "resume.pdf"
    try:
        reader = PdfReader(resume_path)
        pdf_text = ""
        for page in reader.pages:
            pdf_text += page.extract_text() or ""
        print(f"Resume read successfully: {len(pdf_text)} chars")
        return {"resume_text": pdf_text}
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return {"resume_text": ""}

# --- search_for_jobs node ---
def search_for_jobs(state: AgentState):
    """Searches for jobs based on the search_queries in the state."""
    print("--- Searching for Jobs ---")
    
    queries = state.get("search_queries", [])
    if not queries:
        print("No search queries found in state.")
        return {"job_listings": []}

    # --- CHANGE HERE: Explicitly load keys ---
    # try:
        # api_key = os.environ["GOOGLE_API_KEY"]
        # cse_id = os.environ["GOOGLE_CSE_ID"]
        
        # Pass keys directly during instantiation
        # search = GoogleSearchAPIWrapper(google_api_key=api_key, google_cse_id=cse_id)
        
    # except KeyError:
    #     print("ERROR: GOOGLE_API_KEY or GOOGLE_CSE_ID not found in environment variables!")
    #     return {"job_listings": []}
    # --- END CHANGE ---

    try:
        # Instantiate WITHOUT explicitly passing keys
        search = GoogleSearchAPIWrapper() 

    except Exception as e: 
        # Catch potential errors during instantiation if keys are missing/invalid
        print(f"Error initializing GoogleSearchAPIWrapper: {e}")
        return {"job_listings": []}
    
    all_job_listings = []
    
    for query in queries:
        print(f"Searching for: {query}")
        try:
            # Using k=5 to limit results, matching the curl default might be 10
            results = search.results(query, num_results=5) 
            
            for result in results:
                 all_job_listings.append({
                    "snippet": result.get("snippet", "No snippet"),
                    "title": result.get("title", "No title"),
                    "link": result.get("link", "No link")
                })
        except Exception as e:
            # Print the error more clearly
            print(f"Error during search for '{query}': {type(e).__name__} - {e}") 
    
    print(f"Found {len(all_job_listings)} total job listings.")
    
    return {"job_listings": all_job_listings}


# --- NEW NODE: Analyze Jobs ---
def analyze_jobs(state: AgentState):
    """Analyzes each job listing against the resume using an LLM."""
    print("--- Analyzing Job Listings ---")
    
    resume = state.get("resume_text")
    jobs = state.get("job_listings", [])
    jobs = jobs[:1]
    
    if not resume:
        print("Error: Resume text not found in state.")
        return {"analysis_results": []}
    if not jobs:
        print("No job listings found to analyze.")
        return {"analysis_results": []}

    # --- Define the Prompt Template ---
    # We ask the LLM to compare the resume and job description
    # and provide output in a specific JSON format.
    prompt_template = """
    Analyze the provided Resume and Job Listing. Based *only* on the text provided:

    Resume:
    {resume}

    Job Listing:
    Title: {job_title}
    Snippet: {job_snippet}
    Link: {job_link}

    Your Task:
    1. Determine if the resume is a 'High', 'Medium', 'Low', or 'No Match' for the job based *strictly* on the required skills and experience mentioned in the snippet. Do not infer skills the resume might have but doesn't explicitly state.
    2. List the specific skills explicitly required by the job snippet that are NOT mentioned in the resume. If all required skills are present, return an empty list.
    3. State if US or Canadian citizenship is explicitly mentioned as a requirement (e.g., "US Citizen Only", "Canadian Citizen Only"). If not mentioned, use null.
    4. If there are missing skills, provide a brief (1-2 sentence) project idea incorporating one or more of those skills. If no skills are missing, use null.
    5. Extract the posting date if mentioned in the snippet (e.g., "Posted 3 days ago", "2025-10-21"). If not mentioned, use null.

    Provide your analysis ONLY in the following JSON format:
    {format_instructions}
    """

    # --- Setup the Output Parser ---
    # This tells the LLM how to format the JSON and helps us parse it
    parser = JsonOutputParser(pydantic_object=JobAnalysisOutput)

    # --- Create the full LLM Chain ---
    prompt = ChatPromptTemplate.from_template(
        template=prompt_template,
        partial_variables={"format_instructions": parser.get_format_instructions()}
    )
    analyzer_chain = prompt | llm | parser

    # --- Process Each Job ---
    analysis_results_list = []
    print(f"Analyzing {len(jobs)} job listings...")
    for i, job in enumerate(jobs):
        print(f"  Analyzing job {i+1}/{len(jobs)}: {job.get('title', 'N/A')}")
        try:
            # Prepare input for the chain
            chain_input = {
                "resume": resume,
                "job_title": job.get("title", ""),
                "job_snippet": job.get("snippet", ""),
                "job_link": job.get("link", "")
            }
            # Invoke the chain
            raw_result = analyzer_chain.invoke(chain_input)
            
            # The parser gives us a dict matching JobAnalysisOutput
            # Convert Pydantic model back to dict if needed by AgentState TypedDict
            analysis: JobAnalysis = raw_result # Pydantic output matches TypedDict structure here
            analysis_results_list.append(analysis)
            
        except Exception as e:
            print(f"    Error analyzing job {job.get('link', 'N/A')}: {type(e).__name__} - {e}")
            # Optionally add a placeholder error entry
            analysis_results_list.append({
                "job_title": job.get('title', 'Error Processing'),
                "is_match": "Error",
                "missing_skills": [],
                "citizenship_required": None,
                "project_suggestion": f"Error during analysis: {e}",
                "posted_date": None
            })

    print(f"Finished analyzing jobs. {len(analysis_results_list)} results generated.")
    return {"analysis_results": analysis_results_list}


# --- NEW NODE: Compile Report ---
def compile_report(state: AgentState):
    """Compiles the analysis results into a final report string using an LLM."""
    print("--- Compiling Final Report ---")
    
    analysis_results = state.get("analysis_results", [])
    
    if not analysis_results:
        print("No analysis results found to compile report.")
        return {"final_report": "No job analysis results were generated."}

    # --- Define the Prompt Template ---
    # We'll give the LLM the list of analysis results (as a JSON string)
    # and ask it to format them nicely.
    report_prompt_template = """
    Based on the following job analysis results (provided as a JSON list), 
    generate a concise summary report. 

    For each job, include:
    - Job Title
    - Match Level (High, Medium, Low, No Match, Error)
    - Missing Skills (list them or state "None")
    - Citizenship Requirement (if specified)
    - Project Suggestion (if provided)
    - Posted Date (if available)

    Structure the report clearly, perhaps using markdown for headings or bullet points for each job. 
    Start the report with a brief overall summary sentence.

    Analysis Results JSON:
    {analysis_json}

    Final Report:
    """

    # --- Create the Prompt ---
    prompt = ChatPromptTemplate.from_template(report_prompt_template)

    # --- Convert analysis results to a JSON string for the prompt ---
    # We use json.dumps for clean formatting
    analysis_json_string = json.dumps(analysis_results, indent=2)

    # --- Create the Chain (Prompt + LLM + String Output Parser) ---
    # We need a StringOutputParser because we just want the text output
    from langchain_core.output_parsers import StrOutputParser
    
    report_chain = prompt | llm | StrOutputParser()

    # --- Invoke the Chain ---
    try:
        final_report_text = report_chain.invoke({"analysis_json": analysis_json_string})
        print("Report compiled successfully.")
    except Exception as e:
        print(f"Error compiling report: {type(e).__name__} - {e}")
        final_report_text = f"Error generating report: {e}\n\nRaw Analysis:\n{analysis_json_string}"

    # --- Return the Update for the State ---
    return {"final_report": final_report_text}


# ==================================================================
# 4. WIRE UP THE GRAPH
# ==================================================================
# --- Define the graph ---
workflow = StateGraph(AgentState)

# --- Add the nodes ---
# Give each node a unique name and link it to the function
workflow.add_node("resume_reader", read_resume)
workflow.add_node("job_searcher", search_for_jobs)
workflow.add_node("analyzer", analyze_jobs)
workflow.add_node("reporter", compile_report)

# --- Set the entry point ---
workflow.set_entry_point("resume_reader")

# --- Add the edges (our flow is linear) ---
workflow.add_edge("resume_reader", "job_searcher")
workflow.add_edge("job_searcher", "analyzer")
workflow.add_edge("analyzer", "reporter")

# --- The reporter node is the last step, so connect it to the END ---
workflow.add_edge("reporter", END) 

# --- Compile the graph ---
# This creates the executable agent app
app = workflow.compile()


# ==================================================================
# 5. RUN THE GRAPH (Updated Quick Test)
# ==================================================================
if __name__ == "__main__":
    # --- Optional: Comment out for cleaner output ---
    print("\n--- DETAILED ENV VAR CHECK ---")
    print(f"LANGSMITH_TRACING: {os.environ.get('LANGSMITH_TRACING')}")
    print(f"LANGSMITH_ENDPOINT: {os.environ.get('LANGSMITH_ENDPOINT')}")
    api_key_check = os.environ.get('LANGSMITH_API_KEY')
    print(f"LANGSMITH_API_KEY: ...{api_key_check[-5:]}" if api_key_check else "Not Set") 
    print(f"LANGSMITH_PROJECT: {os.environ.get('LANGSMITH_PROJECT')}")
    # print("-----------------------------\n")
    # --- End Optional ---
    # --- Check Env Vars (Keep this for debugging) ---
    print("\n--- DETAILED ENV VAR CHECK ---")
    # ... (keep the env var print checks) ...
    print("-----------------------------\n")

    # --- Define the INPUT for the graph run ---
    # We only need to provide the initial values that aren't generated by a node
    graph_input = {
        "search_queries": [ 
            "Software Architect jobs Canada", 
            "ML Architect jobs USA"
        ]
        # "resume_text" will be filled by the first node
        # "job_listings" will be filled by the second node
        # "analysis_results" by the third, "final_report" by the fourth
    }

    print("--- Running the Agent Graph ---")
    
    # --- Invoke the compiled graph ---
    # LangGraph handles the state updates internally now!
    final_state = app.invoke(graph_input)

    print("\n--- Agent Run Complete ---")
    
    print("\n--- Final Report Output ---")
    # The final report is in the 'final_report' key of the output state
    print(final_state.get("final_report", "Report key not found in final state."))

    # No need for time.sleep() here, LangGraph waits for completion.
    print("\nDone.")


# # --- Updated Quick Test ---
# if __name__ == "__main__":
    
#     # --- UPDATED: We now add our search queries to the initial state ---
#     initial_state = AgentState(
#         resume_text="", 
#         search_queries=[ # <-- Our agent's inputs!
#             "Software Architect jobs Canada", 
#             "ML Architect jobs USA"
#         ], 
#         job_listings=[], 
#         analysis_results=[], 
#         final_report=""
#     )
    
#     # --- Run Node 1: Read Resume ---
#     resume_update = read_resume(initial_state)
#     # Update our state dictionary for the next step
#     current_state = {**initial_state, **resume_update} 
    
#     # --- Run Node 2: Search Jobs ---
#     # Pass the updated state (which includes resume_text now)
#     search_update = search_for_jobs(current_state) 
#     # Update state again
#     current_state = {**current_state, **search_update} 

#     # --- Run Node 3: Analyze Jobs ---
#     analysis_update = analyze_jobs(current_state)
#     print("\n--- Analysis Test Output ---")
#     # Print the list of analysis dictionaries
#     print(json.dumps(analysis_update, indent=2))

#     current_state = {**current_state, **analysis_update} # Update state again


#     # --- Run Node 4: Compile Report ---
#     report_update = compile_report(current_state) # Pass the state including analysis results
#     print("\n--- Final Report Output ---")
#     print(report_update["final_report"]) # Print the generated report string

#     print("\nWaiting 5 seconds for traces to send...")
#     time.sleep(5) 
#     print("Done.")