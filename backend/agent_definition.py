# research_agent/agent_definition.py
import asyncio
import os
import json
import pandas as pd
import uuid
import logging
import time # For timestamps in messages
import re # For parsing URLs if needed
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Optional, Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END

from .tools.web_scraper_tool import WebScraperTool
from .tools.statistical_analyzer_tool import StatisticalAnalyzerTool
from .tools.chart_generator_tool import ChartGeneratorTool

# Setup logging
logger = logging.getLogger(__name__)
if not logger.handlers:
    # Basic console handler if no handlers are configured
    # In a real app, you'd configure this more robustly (e.g., in app_research.py)
    stream_handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s (%(filename)s:%(lineno)d)')
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    logger.setLevel(logging.INFO) # Set to DEBUG for more verbosity

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

llm = None
if GEMINI_API_KEY:
    try:
        llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.3, top_p=0.9) # Adjusted temp/top_p
        logger.info("LLM initialized successfully (gemini-1.5-flash).")
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
else:
    logger.error("GEMINI_API_KEY not found. LLM not initialized. Agent capabilities will be severely limited.")

class ResearchAgentState(TypedDict):
    project_id: str
    user_query: str
    research_plan: str
    search_queries: List[str] # These will be keywords/phrases
    extracted_urls_from_search: List[str] # URLs found from SERP scraping
    scraped_data: Dict[str, str]
    processed_data: Dict[str, str]
    quantitative_data: List[Dict[str, Any]]
    statistical_results: Dict[str, Any]
    qualitative_insights: str
    charts_and_tables: Dict[str, List[str]]
    final_report_markdown: str
    current_project_dir: str
    current_node_message: str # For displaying current agent activity on frontend
    messages: List[Dict[str, Any]] # List of {"timestamp": ..., "text": ... , "type": "info/error/success"}

def _get_llm_text_output(response_text: any, node_name: str, project_id: str) -> str:
    # (Same as previous, ensure it's robust)
    if isinstance(response_text, list):
        if response_text: text_output = str(response_text[0])
        else: text_output = ""
        logger.warning(f"[{project_id}] LLM output for {node_name} was a list, taking first element.")
    elif isinstance(response_text, str): text_output = response_text
    else:
        text_output = str(response_text)
        logger.warning(f"[{project_id}] LLM output for {node_name} was an unexpected type: {type(response_text)}. Casting to string.")
    return text_output.strip()

def safe_write_research_file(filepath: str, content: str, project_id: str):
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info(f"[{project_id}] Successfully wrote research file: {filepath}")
    except Exception as e:
        logger.error(f"[{project_id}] Error writing research file {filepath}: {e}", exc_info=True)
        # Do not re-raise here, just log, as it might be a non-critical save (like intermediate state)
        # The main flow should continue if possible.

def _add_message(state: ResearchAgentState, text: str, msg_type: str = "info") -> List[Dict[str, Any]]:
    """Helper to add a new message to the state's message list."""
    new_message = {"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "text": text, "type": msg_type}
    # Ensure messages list exists and is a list
    current_messages = state.get("messages", [])
    if not isinstance(current_messages, list):
        current_messages = []
    current_messages.append(new_message)
    return current_messages


# --- Agent Nodes (with enhanced logging and state updates) ---

def research_planning_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Research Planning"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 1: Crafting Research Plan & Search Strategy"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    if not llm:
        err_msg = "LLM not available. Skipping research planning."
        logger.error(f"[{project_id}] {err_msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, err_msg, "error"), "current_node_message": "Error: LLM Unavailable"}

    prompt = PromptTemplate.from_template(
        """Given the user's research query: "{user_query}"
        1. Refine this query into a clear, actionable research objective.
        2. Break down the objective into 3-5 key sub-questions or areas of investigation.
        3. For each sub-question, suggest 2-3 specific search query KEYWORDS or PHRASES (not URLs) to use for web research. These should be suitable for a search engine like DuckDuckGo.
        4. Outline a brief research plan.

        Output this as a well-structured Markdown document.
        Then, on a new line, clearly separated by 'SEARCH_QUERIES_JSON_SEPARATOR', provide a JSON list of ONLY the suggested search query keywords/phrases.
        Example JSON list: ["AI in e-commerce customer support", "marketing automation tools for SMBs", "latest AI trends 2024"]
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        logger.info(f"[{project_id}] Invoking LLM for research plan with query: '{state['user_query']}'")
        response_text = chain.invoke({"user_query": state["user_query"]})
        full_output = _get_llm_text_output(response_text, node_name, project_id)
        logger.debug(f"[{project_id}] LLM raw response for plan: {full_output[:500]}...")

        parts = full_output.split("SEARCH_QUERIES_JSON_SEPARATOR")
        plan_markdown = parts[0].strip()
        search_queries_json_str = parts[1].strip() if len(parts) > 1 else "[]"
        
        try:
            search_queries = json.loads(search_queries_json_str)
            if not isinstance(search_queries, list) or not all(isinstance(q, str) for q in search_queries):
                raise ValueError("Parsed JSON is not a list of strings for search queries.")
        except (json.JSONDecodeError, ValueError) as e_json:
            logger.error(f"[{project_id}] Failed to parse search queries JSON: '{search_queries_json_str}'. Error: {e_json}. Falling back to extracting keywords from plan.")
            # Fallback: try to extract some keywords from the plan if JSON fails
            # This is a very basic fallback.
            keywords_from_plan = re.findall(r'"([^"]+)"', plan_markdown) # Look for quoted phrases
            search_queries = list(set(keywords_from_plan))[:5] if keywords_from_plan else ["AI integration trends", state['user_query'][:50]]


        success_msg = f"Research plan generated. {len(search_queries)} search keyword sets identified."
        current_messages = _add_message({"messages": current_messages}, success_msg, "success")
        logger.info(f"[{project_id}] SUCCESS: {success_msg}. Keywords: {search_queries}")
        
        safe_write_research_file(os.path.join(state["current_project_dir"], "1_research_plan.md"), plan_markdown, project_id)
        return {
            **state, 
            "research_plan": plan_markdown, 
            "search_queries": search_queries, 
            "messages": current_messages,
            "current_node_message": current_node_msg + " - Completed"
        }
    except Exception as e:
        error_msg = f"Error in {node_name}: {str(e)}"
        logger.error(f"[{project_id}] {error_msg}", exc_info=True)
        return {**state, "messages": _add_message({"messages": current_messages}, error_msg, "error"), "current_node_message": f"Error in {node_name}"}

def iterative_web_search_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Iterative Web Search & URL Extraction"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 2: Searching Web & Extracting URLs"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    search_keywords_list = state.get("search_queries", [])
    if not search_keywords_list:
        msg = "No search keywords provided from planning phase."
        logger.warning(f"[{project_id}] {msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped (No Keywords)", "extracted_urls_from_search": []}

    scraper = WebScraperTool()
    all_found_urls = []
    max_urls_per_keyword = 2 # Limit how many URLs to get per keyword to avoid too many scrapes

    for i, keywords in enumerate(search_keywords_list):
        keyword_msg = f"Performing web search for keywords ({i+1}/{len(search_keywords_list)}): '{keywords}'"
        current_messages = _add_message({"messages": current_messages}, keyword_msg, "info")
        logger.info(f"[{project_id}] {keyword_msg}")
        
        try:
            # Use the new search_and_extract_urls method
            # This now runs `_scrape_single_url_to_markdown` for the SERP, then parses it.
            extracted_urls = asyncio.run(scraper.search_and_extract_urls(keywords, project_id, max_search_results=max_urls_per_keyword))
            
            if extracted_urls:
                all_found_urls.extend(extracted_urls)
                msg = f"Found {len(extracted_urls)} URLs for '{keywords}': {extracted_urls}"
                current_messages = _add_message({"messages": current_messages}, msg, "info")
                logger.info(f"[{project_id}] {msg}")
            else:
                msg = f"No URLs extracted from search results for keywords: '{keywords}'"
                current_messages = _add_message({"messages": current_messages}, msg, "warning")
                logger.warning(f"[{project_id}] {msg}")
        except Exception as e_search:
            err_msg = f"Error during web search or URL extraction for '{keywords}': {e_search}"
            current_messages = _add_message({"messages": current_messages}, err_msg, "error")
            logger.error(f"[{project_id}] {err_msg}", exc_info=True)
            
    unique_urls = sorted(list(set(all_found_urls))) # Deduplicate and sort
    final_msg = f"URL extraction completed. Found {len(unique_urls)} unique URLs to scrape."
    current_messages = _add_message({"messages": current_messages}, final_msg, "success")
    logger.info(f"[{project_id}] {final_msg} URLs: {unique_urls}")
    
    return {
        **state,
        "extracted_urls_from_search": unique_urls,
        "messages": current_messages,
        "current_node_message": node_name + " - Completed"
    }

def content_scraping_node(state: ResearchAgentState) -> ResearchAgentState: # NEW NODE
    project_id = state['project_id']
    node_name = "Content Scraping"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 2b: Scraping Content from URLs"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    urls_to_scrape = state.get("extracted_urls_from_search", [])
    if not urls_to_scrape:
        msg = "No URLs available for content scraping."
        logger.warning(f"[{project_id}] {msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped (No URLs)"}

    scraper = WebScraperTool()
    # Use the sync wrapper that handles asyncio loop management
    scraped_data_dict = scraper.scrape_urls_sync_via_async_wrapper(urls_to_scrape, project_id)
    
    valid_scraped_data = {}
    processed_data_for_next_node = {} # url -> summary or initial chunk

    if llm: # Prepare summarizer if LLM is available
        summarize_prompt = PromptTemplate.from_template(
            "Concisely summarize the key information from the following text, relevant to the research plan: \nPLAN:\n{research_plan}\n\nURL: {url}\nTEXT (first 8000 chars):\n{text_content}\n\nSUMMARY (2-4 key bullet points or a short paragraph):"
        )
        summarize_chain = summarize_prompt | llm | StrOutputParser()

    for i, url in enumerate(urls_to_scrape):
        content = scraped_data_dict.get(url, "")
        scrape_status_msg = f"Processing scraped content for URL {i+1}/{len(urls_to_scrape)}: {url}"
        current_messages = _add_message({"messages": current_messages}, scrape_status_msg, "info")
        logger.info(f"[{project_id}] {scrape_status_msg}")

        if content and "Error scraping" not in content and "Scrape failed" not in content and content.strip() and "No markdown content found" not in content:
            valid_scraped_data[url] = content
            safe_write_research_file(os.path.join(state["current_project_dir"], f"scraped_content_{i+1}_{url.split('//')[-1].replace('/', '_')[:50]}.md"), content, project_id)
            
            summary_or_fallback = ""
            if llm:
                try:
                    logger.debug(f"[{project_id}] Summarizing content for {url} (length: {len(content)})")
                    summary = summarize_chain.invoke({
                        "url": url, 
                        "research_plan": state.get("research_plan", "General research."),
                        "text_content": content[:8000]
                    })
                    summary_or_fallback = _get_llm_text_output(summary, f"summary_{url}", project_id)
                    logger.debug(f"[{project_id}] Summary for {url}: {summary_or_fallback[:200]}...")
                except Exception as e_sum:
                    logger.error(f"[{project_id}] Error summarizing {url}: {e_sum}", exc_info=True)
                    summary_or_fallback = "Could not summarize. " + content[:500] # Fallback
            else:
                 summary_or_fallback = content[:1000] # Fallback if no LLM for summary
            
            processed_data_for_next_node[url] = summary_or_fallback
            current_messages = _add_message({"messages": current_messages}, f"Successfully scraped and processed: {url}", "success")
        else:
            current_messages = _add_message({"messages": current_messages}, f"Failed to get valid content for: {url}. Detail: {content[:100] if content else 'N/A'}", "warning")
            logger.warning(f"[{project_id}] Failed to get valid content for: {url}. Detail: {content[:100] if content else 'N/A'}")

    final_msg = f"Content scraping completed. {len(valid_scraped_data)} pages successfully scraped and processed out of {len(urls_to_scrape)}."
    current_messages = _add_message({"messages": current_messages}, final_msg, "success")
    logger.info(f"[{project_id}] {final_msg}")
    
    return {
        **state,
        "scraped_data": valid_scraped_data,
        "processed_data": processed_data_for_next_node,
        "messages": current_messages,
        "current_node_message": node_name + " - Completed"
    }


def content_synthesis_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Content Synthesis"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 3: Synthesizing Information"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    if not llm:
        err_msg = "LLM not available. Skipping content synthesis."
        logger.error(f"[{project_id}] {err_msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, err_msg, "error"), "current_node_message": "Error: LLM Unavailable", "qualitative_insights": err_msg}

    processed_content_items = state.get("processed_data", {})
    if not processed_content_items:
        msg = "No processed content to synthesize."
        logger.warning(f"[{project_id}] {msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped", "qualitative_insights": msg}

    full_processed_text = ""
    for url, summary_or_chunk in processed_content_items.items():
        full_processed_text += f"Source URL: {url}\nContent/Summary:\n{summary_or_chunk}\n\n{'='*40}\n\n"
    
    if not full_processed_text.strip():
        msg = "Processed content is empty after concatenation, cannot synthesize."
        logger.warning(f"[{project_id}] {msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped", "qualitative_insights": msg}

    prompt = PromptTemplate.from_template(
        """Based on the research plan:
        {research_plan}

        And the following collection of processed information (summaries or initial chunks) from multiple web sources:
        {processed_texts}

        Synthesize these findings into a coherent qualitative analysis. Identify:
        1. Key themes, main arguments, and recurring ideas.
        2. Supporting evidence or data points mentioned (describe them qualitatively).
        3. Any conflicting information, different perspectives, or nuances.
        4. Potential gaps in the information gathered so far relevant to the research plan.
        
        Provide a comprehensive qualitative analysis as a well-structured Markdown document.
        Focus on clarity and actionable insights for the user.
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        max_synthesis_context = 25000 # Gemini 1.5 Flash can handle more
        synthesis_input_text = full_processed_text
        if len(synthesis_input_text) > max_synthesis_context:
            synthesis_input_text = synthesis_input_text[:max_synthesis_context]
            warn_msg = f"Warning: Truncated processed text for synthesis due to context length ({len(full_processed_text)} -> {max_synthesis_context} chars)."
            current_messages = _add_message({"messages": current_messages}, warn_msg, "warning")
            logger.warning(f"[{project_id}] {warn_msg}")
        
        logger.info(f"[{project_id}] Invoking LLM for content synthesis (input length: {len(synthesis_input_text)} chars).")
        response_text = chain.invoke({
            "research_plan": state.get("research_plan", "General research objective."),
            "processed_texts": synthesis_input_text
        })
        insights = _get_llm_text_output(response_text, node_name, project_id)
        logger.debug(f"[{project_id}] LLM raw response for synthesis: {insights[:500]}...")
        
        success_msg = "Content synthesis completed."
        current_messages = _add_message({"messages": current_messages}, success_msg, "success")
        logger.info(f"[{project_id}] SUCCESS: {success_msg}")
        safe_write_research_file(os.path.join(state["current_project_dir"], "2_qualitative_synthesis.md"), insights, project_id)
        return {**state, "qualitative_insights": insights, "messages": current_messages, "current_node_message": current_node_msg + " - Completed"}
    except Exception as e:
        error_msg = f"Error in {node_name}: {str(e)}"
        logger.error(f"[{project_id}] {error_msg}", exc_info=True)
        return {**state, "messages": _add_message({"messages": current_messages}, error_msg, "error"), "current_node_message": f"Error in {node_name}", "qualitative_insights": f"Error during synthesis: {e}"}

def quantitative_extraction_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Quantitative Data Extraction"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 4: Extracting Quantitative Data"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    if not llm:
        err_msg = "LLM not available. Skipping quantitative extraction."
        logger.error(f"[{project_id}] {err_msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, err_msg, "error"), "current_node_message": "Error: LLM Unavailable", "quantitative_data": []}

    text_to_extract_from = state.get("qualitative_insights", "")
    if not text_to_extract_from.strip() or len(text_to_extract_from) < 100:
        logger.warning(f"[{project_id}] Qualitative insights seem too short or empty. Combining all processed data for quantitative extraction.")
        combined_processed = "\n\n---\n\n".join(state.get("processed_data", {}).values())
        if combined_processed.strip():
            text_to_extract_from = combined_processed
            current_messages = _add_message({"messages": current_messages}, "Using combined processed data for quantitative extraction as synthesis was short/empty.", "info")
        else:
            msg = "No text available (neither synthesis nor processed data) for quantitative data extraction."
            logger.warning(f"[{project_id}] {msg}")
            return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped", "quantitative_data": []}
            
    max_extraction_context = 20000
    if len(text_to_extract_from) > max_extraction_context:
        text_to_extract_from = text_to_extract_from[:max_extraction_context]
        warn_msg = f"Warning: Truncated text for quantitative extraction ({len(text_to_extract_from)} -> {max_extraction_context} chars)."
        current_messages = _add_message({"messages": current_messages}, warn_msg, "warning")
        logger.warning(f"[{project_id}] {warn_msg}")

    prompt = PromptTemplate.from_template(
        """From the following text, extract any quantitative data points, statistics, or numerical facts relevant to the research plan:
        Research Plan Context:
        {research_plan}

        Text to Analyze:
        {text_data}

        Extract the data as a JSON list of objects. Each object should have AT LEAST the following keys:
        - "metric_name": A descriptive name for the data point (e.g., "Market Size", "Growth Rate", "User Count", "Price Point").
        - "value": The numerical value (MUST be a float or int, parse it from text like '$1.5M' to 1500000.0 or '25%' to 25.0). If a range like "10-15", you can pick midpoint or represent as "10-15".
        - "unit": The unit of the value (e.g., "USD", "Million USD", "%", "Users", "YoY", "Per Month"). If no unit, use "N/A" or infer if obvious (e.g. count).
        - "context_notes": Brief context, source hint if identifiable, or the sentence from which it was extracted (e.g., "Source: XYZ Report 2023", "Q1 Figure for Product A", "As stated in the introduction...").

        Example:
        [
            {{"metric_name": "AI Market Size", "value": 150200000.0, "unit": "USD", "context_notes": "Projected for 2024, mentioned as $150.2 Billion"}},
            {{"metric_name": "Adoption Rate for SMBs", "value": 25.0, "unit": "%", "context_notes": "Among Small and Medium Businesses"}}
        ]
        If no quantitative data is found, return an empty list [].
        Output ONLY the JSON list. Ensure the JSON is perfectly valid.
        """
    )
    json_parser = JsonOutputParser()
    chain = prompt | llm | json_parser

    try:
        logger.info(f"[{project_id}] Invoking LLM for quantitative data extraction (input length: {len(text_to_extract_from)} chars).")
        # The JsonOutputParser will parse the LLM's string output into a Python list/dict
        extracted_data_list_raw = chain.invoke({
            "research_plan": state.get("research_plan", "General research."),
            "text_data": text_to_extract_from
        })
        logger.debug(f"[{project_id}] LLM raw response for quantitative data (pre-JsonOutputParser if direct string): {str(extracted_data_list_raw)[:500]}...")


        # Validate and clean the output from JsonOutputParser
        final_extracted_data = []
        if isinstance(extracted_data_list_raw, list):
            for item in extracted_data_list_raw:
                if isinstance(item, dict) and "metric_name" in item and "value" in item:
                    # Attempt to coerce 'value' to numeric if it's a string that looks numeric
                    if isinstance(item["value"], str):
                        try:
                            # Basic cleaning for common currency/percentage strings
                            cleaned_val_str = item["value"].replace('$', '').replace(',', '').replace('%', '').strip()
                            if cleaned_val_str.lower() not in ['n/a', 'na', '', '-']:
                                item["value"] = float(cleaned_val_str)
                            else: # If it's explicitly N/A, skip or mark
                                continue # Or item["value"] = None, depending on how you want to handle
                        except ValueError:
                            logger.warning(f"[{project_id}] Could not convert value '{item['value']}' to float for metric '{item['metric_name']}'. Keeping as string or skipping.")
                            # Decide: skip item, keep as string, or set value to None
                            # For now, let's keep it if it was a string, stats tool will try to parse again
                            pass 
                    final_extracted_data.append(item)
                else:
                    logger.warning(f"[{project_id}] Skipping invalid item in quantitative data list: {item}")
            state["quantitative_data"] = final_extracted_data
        else:
            logger.error(f"[{project_id}] Quantitative extraction did not return a list as expected from JsonOutputParser. Got: {type(extracted_data_list_raw)}. Data: {str(extracted_data_list_raw)[:200]}")
            state["quantitative_data"] = []
            
        success_msg = f"Quantitative data extraction completed. Found {len(state['quantitative_data'])} valid data points."
        current_messages = _add_message({"messages": current_messages}, success_msg, "success")
        logger.info(f"[{project_id}] SUCCESS: {success_msg}. Data: {state['quantitative_data'][:3]}") # Log first 3 items
        safe_write_research_file(os.path.join(state["current_project_dir"], "3_quantitative_data.json"), json.dumps(state["quantitative_data"], indent=2), project_id)
        return {**state, "messages": current_messages, "current_node_message": current_node_msg + " - Completed"}

    except Exception as e: # Catches JSON parsing errors from JsonOutputParser or other issues
        error_msg = f"Error in {node_name}: {str(e)}"
        logger.error(f"[{project_id}] {error_msg}", exc_info=True)
        # Log the raw response if parsing failed and it's available
        raw_resp_for_log = ""
        if 'extracted_data_list_raw' in locals() and isinstance(extracted_data_list_raw, str): 
            raw_resp_for_log = extracted_data_list_raw[:500]
        logger.error(f"[{project_id}] Raw LLM response leading to parse error (first 500 chars): {raw_resp_for_log}")
        
        return {**state, "messages": _add_message({"messages": current_messages}, error_msg, "error"), "current_node_message": f"Error in {node_name}", "quantitative_data": []}


def statistical_analysis_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Statistical Analysis"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 5: Performing Statistical Analysis"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    quantitative_data = state.get("quantitative_data", [])
    if not quantitative_data:
        msg = "No quantitative data available for statistical analysis."
        logger.warning(f"[{project_id}] {msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped", "statistical_results": {"summary": msg}}

    analyzer = StatisticalAnalyzerTool()
    results: Dict[str, Any] = {"summary_text": "Statistical Analysis Results:", "descriptive_stats_list": []}
    
    df = pd.DataFrame(quantitative_data)
    all_stats_text_parts = []

    # Group by 'metric_name' to analyze each metric's set of 'value's
    if 'metric_name' in df.columns and 'value' in df.columns:
        grouped_by_metric = df.groupby('metric_name')
        logger.info(f"[{project_id}] Found {len(grouped_by_metric)} unique metric groups for statistical analysis.")

        for metric_name, group_df in grouped_by_metric:
            logger.info(f"[{project_id}] Analyzing metric group: '{metric_name}' with {len(group_df)} entries.")
            # The tool expects List[Dict], and the column to analyze ('value')
            metric_specific_data_list_of_dicts = group_df.to_dict('records')
            
            stats = analyzer.calculate_descriptive_stats(metric_specific_data_list_of_dicts, 'value', project_id)
            
            if stats and "error" not in stats:
                stats['metric_name_analyzed'] = metric_name # Add context for which metric this stat belongs to
                results["descriptive_stats_list"].append(stats)
                all_stats_text_parts.append(analyzer.format_stats_as_text(stats, project_id))
                logger.info(f"[{project_id}] Calculated stats for '{metric_name}'.")
            elif stats and "error" in stats:
                err_msg_stat = f"Error analyzing '{metric_name}': {stats['error']}"
                all_stats_text_parts.append(err_msg_stat)
                logger.warning(f"[{project_id}] {err_msg_stat}")
            else:
                logger.warning(f"[{project_id}] No stats returned or unexpected result for metric '{metric_name}'.")
    else:
        logger.warning(f"[{project_id}] 'metric_name' or 'value' columns not found in quantitative data for grouped analysis.")
        # Try to analyze 'value' column if it exists globally (less ideal)
        if 'value' in df.columns:
            stats = analyzer.calculate_descriptive_stats(df.to_dict('records'), 'value', project_id)
            if stats and "error" not in stats:
                stats['metric_name_analyzed'] = "Overall Values"
                results["descriptive_stats_list"].append(stats)
                all_stats_text_parts.append(analyzer.format_stats_as_text(stats, project_id))
            elif stats and "error" in stats:
                 all_stats_text_parts.append(f"Error analyzing overall 'value' column: {stats['error']}")


    if all_stats_text_parts:
        results["summary_text"] += "\n\n" + "\n\n".join(all_stats_text_parts)
    else:
        no_stats_msg = "\nNo suitable numeric data found or processed for detailed descriptive statistics."
        results["summary_text"] += no_stats_msg
        logger.info(f"[{project_id}] {no_stats_msg.strip()}")
    
    success_msg = "Statistical analysis completed."
    current_messages = _add_message({"messages": current_messages}, success_msg, "success")
    logger.info(f"[{project_id}] SUCCESS: {success_msg}")
    safe_write_research_file(os.path.join(state["current_project_dir"], "4_statistical_results.json"), json.dumps(results, indent=2), project_id)
    return {**state, "statistical_results": results, "messages": current_messages, "current_node_message": current_node_msg + " - Completed"}

def visualization_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Visualization Generation"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 6: Generating Charts & Tables"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    charts_output_dir = os.path.join(state["current_project_dir"], "charts") # Charts will be saved here
    chart_tool = ChartGeneratorTool(output_dir=charts_output_dir, project_id=project_id)

    quantitative_data = state.get("quantitative_data", [])
    statistical_results_dict = state.get("statistical_results", {}) # This now contains 'descriptive_stats_list'
    
    generated_visuals: Dict[str, List[str]] = {"charts": [], "tables_md": []}

    if not quantitative_data:
        msg = "No quantitative data available for creating visualizations."
        logger.warning(f"[{project_id}] {msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, msg, "warning"), "current_node_message": node_name + " - Skipped", "charts_and_tables": generated_visuals}

    df_quant = pd.DataFrame(quantitative_data)

    # 1. Markdown table from all quantitative data
    if not df_quant.empty:
        table_title = "Extracted Quantitative Data Overview"
        generated_visuals["tables_md"].append(chart_tool.generate_markdown_table(quantitative_data, title=table_title, project_id=project_id))
        current_messages = _add_message({"messages": current_messages}, f"Generated table: {table_title}", "info")

    # 2. Bar chart for metrics (if 'metric_name' and 'value' exist and 'value' is numeric)
    if 'metric_name' in df_quant.columns and 'value' in df_quant.columns:
        try:
            # Ensure 'value' is numeric for plotting
            df_quant['value_numeric'] = pd.to_numeric(df_quant['value'], errors='coerce')
            df_plot_bar = df_quant.dropna(subset=['value_numeric', 'metric_name'])
            
            if not df_plot_bar.empty:
                # Group by metric_name and take the mean if multiple entries, or sum, or first.
                # For simplicity, if a metric appears multiple times, let's average its value for the bar chart.
                # Or, if the LLM was instructed to provide unique metrics, this might not be needed.
                # Assuming for now that 'metric_name' might have multiple 'value' entries.
                # Let's plot each metric if not too many, or top N.
                
                # Example: Bar chart of values by metric_name (if few unique metrics)
                unique_metrics = df_plot_bar['metric_name'].unique()
                logger.info(f"[{project_id}] Unique metrics for bar chart consideration: {len(unique_metrics)}")
                if 0 < len(unique_metrics) <= 15: # Plot if a reasonable number of unique metrics
                    # If values are already aggregated per metric_name by LLM, this is fine.
                    # If not, you might want to df_plot_bar.groupby('metric_name')['value_numeric'].mean().reset_index()
                    chart_path_bar = chart_tool.generate_bar_chart(
                        df_plot_bar.to_dict('records'), 
                        category_col='metric_name', 
                        value_col='value_numeric', 
                        title="Key Metrics Overview"
                    )
                    if chart_path_bar:
                        generated_visuals["charts"].append(os.path.relpath(chart_path_bar, state["current_project_dir"]))
                        current_messages = _add_message({"messages": current_messages}, f"Generated bar chart: Key Metrics Overview", "success")
                else:
                    logger.warning(f"[{project_id}] Too many unique metrics ({len(unique_metrics)}) for a single bar chart. Consider alternative visualization or data aggregation.")
                    current_messages = _add_message({"messages": current_messages}, f"Skipped combined bar chart due to too many ({len(unique_metrics)}) unique metrics.", "warning")
            else:
                logger.warning(f"[{project_id}] No numeric 'value' data found for bar chart after cleaning.")
        except Exception as e_bar:
            logger.error(f"[{project_id}] Error preparing or generating bar chart: {e_bar}", exc_info=True)
            current_messages = _add_message({"messages": current_messages}, f"Error generating bar chart: {e_bar}", "error")

    # 3. Table from statistical_results (descriptive_stats_list part)
    descriptive_stats_list = statistical_results_dict.get("descriptive_stats_list", [])
    if descriptive_stats_list:
        stats_data_for_table = []
        for item_stats in descriptive_stats_list:
            flat_stat = {"Metric Analyzed": item_stats.get("metric_name_analyzed", item_stats.get("column", "N/A"))}
            for k,v_stat in item_stats.items():
                if k not in ["metric_name_analyzed", "column", "error"]: # Exclude error key from table
                    flat_stat[k.replace('_', ' ').title()] = f"{v_stat:.2f}" if isinstance(v_stat, (float, int)) else v_stat
            stats_data_for_table.append(flat_stat)
        
        if stats_data_for_table:
            table_title_stats = "Descriptive Statistics Summary"
            generated_visuals["tables_md"].append(chart_tool.generate_markdown_table(stats_data_for_table, title=table_title_stats, project_id=project_id))
            current_messages = _add_message({"messages": current_messages}, f"Generated table: {table_title_stats}", "info")

    success_msg = f"Visualization generation completed. {len(generated_visuals['charts'])} charts, {len(generated_visuals['tables_md'])} tables."
    current_messages = _add_message({"messages": current_messages}, success_msg, "success")
    logger.info(f"[{project_id}] SUCCESS: {success_msg}")
    
    return {**state, "charts_and_tables": generated_visuals, "messages": current_messages, "current_node_message": current_node_msg + " - Completed"}

def report_compilation_node(state: ResearchAgentState) -> ResearchAgentState:
    project_id = state['project_id']
    node_name = "Final Report Compilation"
    current_messages = _add_message(state, f"Starting: {node_name}...", "info")
    current_node_msg = f"Phase 7: Compiling Final Research Report"
    logger.info(f"[{project_id}] ENTERING NODE: {node_name}")

    if not llm:
        err_msg = "LLM not available. Skipping final report compilation."
        logger.error(f"[{project_id}] {err_msg}")
        return {**state, "messages": _add_message({"messages": current_messages}, err_msg, "error"), "current_node_message": "Error: LLM Unavailable", "final_report_markdown": f"# Report Compilation Failed\n{err_msg}"}

    plan = state.get("research_plan", "No research plan was generated or available.")
    qual_insights = state.get("qualitative_insights", "No qualitative insights were generated or available.")
    stats_results_dict = state.get("statistical_results", {})
    # Use the 'summary_text' which should now contain formatted stats or error messages
    stats_summary_text = stats_results_dict.get("summary_text", "No statistical analysis summary available.")
    
    charts_and_tables_dict = state.get("charts_and_tables", {"charts": [], "tables_md": []})
    table_markdowns_combined = "\n\n".join(charts_and_tables_dict.get("tables_md", ["No tables were generated."]))
    
    chart_markdowns_list = []
    for chart_rel_path in charts_and_tables_dict.get("charts", []):
        chart_filename = os.path.basename(chart_rel_path)
        chart_title = chart_filename.replace('_', ' ').replace('.png', '').title()
        # Ensure the path in markdown is just the filename if charts are in a subdir of the report's location
        # Or use the relative path from project root if the markdown viewer can handle it.
        # For simplicity, assuming charts are in "charts/" subdir relative to the report.md
        # The chart_rel_path is already like "charts/image.png"
        chart_markdowns_list.append(f"![{chart_title}]({chart_rel_path})") 
    charts_section_md_combined = "\n\n".join(chart_markdowns_list) if chart_markdowns_list else "No charts were generated or an error occurred."

    prompt = PromptTemplate.from_template(
        """Compile a comprehensive and professional research report based on the following sections.
        The report should be well-structured in Markdown format, suitable for a solopreneur/freelancer specializing in AI integration.
        Focus on clarity, actionable insights, and a polished presentation.

        User's Original Research Query: {user_query}

        ## 1. Executive Summary
        (Craft a concise overview [2-3 paragraphs] of the most critical findings, key trends, and main conclusions from the entire research. This should be a stand-alone summary.)

        ## 2. Research Plan & Methodology
        (Briefly restate the core research objectives derived from the user's query. Outline the methodology: e.g., keyword-based web searches on DuckDuckGo, content scraping, LLM-based synthesis and data extraction, statistical summarization, and visualization.)
        Original Research Plan Outline:
        ```markdown
        {research_plan}
        ```

        ## 3. Key Qualitative Insights & Thematic Analysis
        (Present the synthesized qualitative findings from the web research. Organize by themes. Use bullet points or short paragraphs for clarity. Highlight any surprising or particularly relevant insights.)
        ```markdown
        {qualitative_insights}
        ```

        ## 4. Quantitative Data Overview & Statistical Highlights
        (Present the summary of statistical analysis performed. Directly embed the Markdown tables generated previously. If specific data points are crucial, mention them in a brief narrative before or after the tables.)
        Statistical Analysis Summary & Tables:
        {statistical_summary_and_tables_markdown}

        ## 5. Visualizations
        (Embed the generated charts here. Provide a brief interpretation or caption for each chart if not already part of the image title. Ensure image paths like `charts/chart_name.png` are used.)
        {charts_section_markdown}

        ## 6. Conclusions & Potential Next Steps
        (Summarize the overall conclusions drawn from all aspects of the research. Based on the findings, suggest potential implications, opportunities, or areas for further investigation relevant to an AI integration specialist.)

        ---
        *Report generated by AI Research Agent.*

        Ensure the report flows logically, is easy to read, and uses professional language. Use Markdown formatting effectively (headings, subheadings, lists, bolding, blockquotes for important notes if any).
        """
    )
    chain = prompt | llm | StrOutputParser()
    try:
        logger.info(f"[{project_id}] Invoking LLM for final report compilation.")
        # Combine stats summary text and markdown tables for the prompt
        full_stats_and_tables_md = f"{stats_summary_text}\n\n{table_markdowns_combined}"

        response_text = chain.invoke({
            "user_query": state["user_query"],
            "research_plan": plan,
            "qualitative_insights": qual_insights,
            "statistical_summary_and_tables_markdown": full_stats_and_tables_md,
            "charts_section_markdown": charts_section_md_combined
        })
        final_report = _get_llm_text_output(response_text, node_name, project_id)
        logger.debug(f"[{project_id}] LLM raw response for final report: {final_report[:500]}...")
        
        success_msg = "Final research report compiled successfully."
        current_messages = _add_message({"messages": current_messages}, success_msg, "success")
        logger.info(f"[{project_id}] SUCCESS: {success_msg}")
        safe_write_research_file(os.path.join(state["current_project_dir"], "final_research_report.md"), final_report, project_id)
        return {**state, "final_report_markdown": final_report, "messages": current_messages, "current_node_message": current_node_msg + " - Completed"}
    except Exception as e:
        error_msg = f"Error in {node_name}: {str(e)}"
        logger.error(f"[{project_id}] {error_msg}", exc_info=True)
        final_report_error_content = f"# Report Compilation Failed\n\nAn error occurred during the final report generation: {e}\n\n"
        final_report_error_content += "## Partially Available Information:\n\n"
        final_report_error_content += f"### Research Plan:\n{plan}\n\n"
        final_report_error_content += f"### Qualitative Insights:\n{qual_insights}\n\n"
        final_report_error_content += f"### Statistical Summary & Tables:\n{stats_summary_text}\n{table_markdowns_combined}\n\n"
        final_report_error_content += f"### Charts Section (Paths):\n{charts_section_md_combined}\n"
        safe_write_research_file(os.path.join(state["current_project_dir"], "final_research_report_ERROR.md"), final_report_error_content, project_id)
        return {**state, "messages": _add_message({"messages": current_messages}, error_msg, "error"), "current_node_message": f"Error in {node_name}", "final_report_markdown": final_report_error_content}


# --- Graph Definition ---
graph_builder = StateGraph(ResearchAgentState)

graph_builder.add_node("research_planning", research_planning_node)
graph_builder.add_node("iterative_web_search", iterative_web_search_node) # This node now only gets URLs
graph_builder.add_node("content_scraping", content_scraping_node) # New node for actual scraping
graph_builder.add_node("content_synthesis", content_synthesis_node)
graph_builder.add_node("quantitative_extraction", quantitative_extraction_node)
graph_builder.add_node("statistical_analysis", statistical_analysis_node)
graph_builder.add_node("visualization", visualization_node)
graph_builder.add_node("report_compilation", report_compilation_node)

graph_builder.set_entry_point("research_planning")
graph_builder.add_edge("research_planning", "iterative_web_search")
graph_builder.add_edge("iterative_web_search", "content_scraping") # Search -> Scrape
graph_builder.add_edge("content_scraping", "content_synthesis")   # Scrape -> Synthesize
graph_builder.add_edge("content_synthesis", "quantitative_extraction")
graph_builder.add_edge("quantitative_extraction", "statistical_analysis")
graph_builder.add_edge("statistical_analysis", "visualization")
graph_builder.add_edge("visualization", "report_compilation")
graph_builder.add_edge("report_compilation", END)

research_graph = None
if llm:
    try:
        research_graph = graph_builder.compile()
        logger.info("Research agent graph compiled successfully.")
    except Exception as e:
        logger.error(f"Failed to compile research agent graph: {e}", exc_info=True)
else:
    logger.error("Research agent graph not compiled as LLM is unavailable.")

def run_research_agent(user_query: str, project_id: Optional[str] = None) -> ResearchAgentState:
    # (Ensure this function initializes all new fields in AgentState, especially `extracted_urls_from_search`)
    if not project_id:
        project_id = f"research_{uuid.uuid4()}"
    
    current_project_dir = os.path.abspath(os.path.join("research_projects", project_id))
    os.makedirs(current_project_dir, exist_ok=True) # Ensure it exists

    initial_timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    initial_messages = [{"timestamp": initial_timestamp, "text": f"Research agent initialized for Project ID: {project_id}", "type": "system"}]
    initial_messages.append({"timestamp": initial_timestamp, "text": f"User Query: '{user_query}'", "type": "info"})


    initial_state = ResearchAgentState(
        project_id=project_id,
        user_query=user_query,
        research_plan="",
        search_queries=[], # Will be populated by planning node
        extracted_urls_from_search=[], # Will be populated by web_search node
        scraped_data={},
        processed_data={},
        quantitative_data=[],
        statistical_results={},
        qualitative_insights="",
        charts_and_tables={"charts": [], "tables_md": []},
        final_report_markdown="",
        current_project_dir=current_project_dir,
        current_node_message="Initializing research agent...",
        messages=initial_messages
    )

    if not research_graph: # Handles case where LLM failed to init or graph compilation failed
        error_message = "Research graph is not compiled (LLM might be unavailable or graph compilation failed). Cannot run agent flow."
        logger.error(f"[{project_id}] {error_message}")
        initial_state["messages"] = _add_message(initial_state, error_message, "error")
        initial_state["current_node_message"] = "Agent Error"
        initial_state["final_report_markdown"] = f"# Agent System Error\n\n{error_message}"
        try:
            safe_write_research_file(os.path.join(current_project_dir, "state.json"), json.dumps(initial_state, indent=2, default=str), project_id) # Added default=str for any non-serializable types
            safe_write_research_file(os.path.join(current_project_dir, "final_research_report_ERROR.md"), initial_state["final_report_markdown"], project_id)
        except Exception as e_save:
            logger.error(f"[{project_id}] Error saving error state: {e_save}", exc_info=True)
        return initial_state
    
    logger.info(f"[{project_id}] Starting research agent flow with initial state.")
    config = {"recursion_limit": 35} 

    # Save initial state before starting
    state_file_path = os.path.join(current_project_dir, "state.json")
    try:
        safe_write_research_file(state_file_path, json.dumps(initial_state, indent=2, default=str), project_id)
    except Exception as e_init_save:
        logger.error(f"[{project_id}] Could not write initial state.json: {e_init_save}", exc_info=True)
        # Continue if possible, but log this failure

    final_state = research_graph.invoke(initial_state, config=config)
    
    logger.info(f"[{project_id}] Research agent flow completed. Final node message: {final_state.get('current_node_message')}")
    try:
        safe_write_research_file(state_file_path, json.dumps(final_state, indent=2, default=str), project_id)
        logger.info(f"[{project_id}] Final state saved to state.json.")
    except Exception as e_final_save:
        logger.error(f"[{project_id}] Could not write final state.json: {e_final_save}", exc_info=True)
        final_state["messages"] = _add_message(final_state, f"Warning: Could not save final state to file: {e_final_save}", "warning")
        
    return final_state

# (Keep if __name__ == '__main__': block for testing agent_definition.py directly)
if __name__ == '__main__':
    if not llm:
        print("LLM not initialized. Exiting example research agent run.")
    else:
        print("Starting example research agent run from agent_definition.py...")
        # test_query = "What are the current trends in AI integration for small e-commerce businesses in 2024, focusing on customer support and marketing automation?"
        test_query = "Market size and growth projections for AI in healthcare diagnostics in North America for 2024-2026."
        
        final_output = run_research_agent(test_query)
        
        print("\n--- Agent Messages Log ---")
        for msg_obj in final_output.get("messages", []):
            print(f"[{msg_obj.get('timestamp', 'N/A')}] [{msg_obj.get('type', 'INFO').upper()}] {msg_obj.get('text', '')}")
        
        print(f"\n--- Final Report for Project ID: {final_output['project_id']} ---")
        print(final_output.get("final_report_markdown", "No final report generated or an error occurred."))
        
        report_path = os.path.join(final_output["current_project_dir"], "final_research_report.md")
        print(f"\nFull report should be saved to: {report_path}")
        
        charts = final_output.get("charts_and_tables", {}).get("charts", [])
        if charts:
            print("\nGenerated Charts (relative to project dir):")
            for chart_file in charts:
                print(f"- {chart_file} (Full path: {os.path.join(final_output['current_project_dir'], chart_file)})")