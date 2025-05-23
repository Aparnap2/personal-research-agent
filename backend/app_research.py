# app_research.py
from flask import Flask, request, jsonify, send_from_directory, url_for
from flask_cors import CORS
import os
import json
import logging
import time # For initial message timestamp
import agent_definition
import database # Import our SQLite database module

# Configure basic logging for Flask app
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - [%(threadName)s] - %(message)s (%(filename)s:%(lineno)d)')
flask_logger = logging.getLogger('flask.research_app')
# Mute werkzeug INFO logs if too noisy, but keep warnings/errors
logging.getLogger('werkzeug').setLevel(logging.WARNING)


app = Flask(__name__, static_folder='../frontend/dist', static_url_path='/')

# Robust CORS for development (allows Vite dev server) and potentially production
CORS(app, resources={r"/*": {
    "origins": ["http://localhost:5173", "http://127.0.0.1:5173"], # Add your frontend dev origin(s)
    "methods": ["GET", "POST", "OPTIONS"],
    "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
    "supports_credentials": True # If you ever use cookies/sessions
}})

app.config['RESEARCH_PROJECTS_DIR'] = os.path.abspath(os.path.join(os.getcwd(), "research_projects"))

if not os.path.exists(app.config['RESEARCH_PROJECTS_DIR']):
    try:
        os.makedirs(app.config['RESEARCH_PROJECTS_DIR'], mode=0o700)
        flask_logger.info(f"Created research projects directory: {app.config['RESEARCH_PROJECTS_DIR']}")
    except Exception as e:
        flask_logger.error(f"Failed to create research projects directory: {e}", exc_info=True)


@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/api/start_research', methods=['POST']) # OPTIONS handled by Flask-CORS
def start_research_route():
    flask_logger.debug(f"API: Received {request.method} for /api/start_research. Headers: {request.headers}")
    if not request.is_json:
        flask_logger.warning("API: Start research request content type not application/json.")
        return jsonify({"error": "Request body must be JSON."}), 415 # Unsupported Media Type
        
    data = request.get_json()
    user_query = data.get('query')

    if not user_query:
        flask_logger.warning("API: Start research request missing 'query' in JSON body.")
        return jsonify({"error": "Research 'query' is required in JSON body."}), 400
    
    flask_logger.info(f"API: Received research request. Query: '{user_query[:100]}...'") # Log snippet
    try:
        # This is a synchronous call. For production, use a task queue (Celery, RQ).
        # The agent itself will run its graph.
        initial_agent_state = agent_definition.run_research_agent(user_query=user_query)
        
        project_id = initial_agent_state["project_id"]
        flask_logger.info(f"API: Agent process initiated for project_id: {project_id}. Initial messages count: {len(initial_agent_state.get('messages',[]))}")
        
        return jsonify({
            "message": "Research agent process started.",
            "project_id": project_id,
            "status_url": url_for('get_research_status_route', project_id=project_id, _external=False) # Use relative for frontend proxy
        }), 202
    except Exception as e:
        flask_logger.error(f"API: Critical error starting research agent: {str(e)}", exc_info=True)
        return jsonify({"error": f"An unexpected server error occurred while starting agent: {str(e)}"}), 500

@app.route('/api/research_status/<project_id>', methods=['GET']) # OPTIONS handled by Flask-CORS
def get_research_status_route(project_id):
    flask_logger.debug(f"API: Received GET for /api/research_status/{project_id}")
    project_dir = os.path.join(app.config['RESEARCH_PROJECTS_DIR'], project_id)
    state_file_path = os.path.join(project_dir, "state.json")
    report_file_path = os.path.join(project_dir, "final_research_report.md") # Main report
    error_report_file_path = os.path.join(project_dir, "final_research_report_ERROR.md") # Error report

    if not os.path.exists(project_dir):
        flask_logger.warning(f"API: Project directory not found for status check: {project_id}")
        return jsonify({"error": "Project not found."}), 404

    status_response = {
        "project_id": project_id, 
        "completed": False, 
        "messages": [{"timestamp": time.strftime('%Y-%m-%d %H:%M:%S'), "text": "Awaiting agent state...", "type": "system"}], 
        "current_node_message": "Initializing...", 
        "report_markdown": None, 
        "charts": [],
        "error_message": None
    }
    
    if os.path.exists(state_file_path):
        try:
            with open(state_file_path, "r", encoding="utf-8") as f:
                agent_state = json.load(f)
            
            status_response["messages"] = agent_state.get("messages", status_response["messages"])
            status_response["current_node_message"] = agent_state.get("current_node_message", status_response["current_node_message"])
            
            # Check for completion based on final_report_markdown presence or error state
            if agent_state.get("final_report_markdown"):
                if os.path.exists(report_file_path) and "Error" not in agent_state.get("final_report_markdown", ""):
                    status_response["completed"] = True
                    status_response["report_markdown"] = agent_state["final_report_markdown"]
                    charts_info = agent_state.get("charts_and_tables", {}).get("charts", [])
                    status_response["charts"] = [
                        {"title": os.path.basename(p).replace('_', ' ').replace('.png','').title(), 
                         "url": url_for('get_research_chart_route', project_id=project_id, filename=os.path.basename(p), _external=False)} # Relative for proxy
                        for p in charts_info
                    ]
                elif os.path.exists(error_report_file_path) or "Error" in agent_state.get("final_report_markdown", ""):
                    status_response["completed"] = True # Completed with error
                    status_response["error_message"] = "Research process encountered an error. Check logs or partial report."
                    status_response["report_markdown"] = agent_state.get("final_report_markdown") # Show error report
                    flask_logger.warning(f"API: Research for {project_id} completed with an error state.")


            # Fallback error check from messages if not explicitly in final_report_markdown
            if not status_response["completed"] and any(msg.get("type") == "error" for msg in status_response["messages"]):
                status_response["completed"] = True # If any error message, consider it "done" for polling
                status_response["error_message"] = next((msg["text"] for msg in reversed(status_response["messages"]) if msg.get("type") == "error"), "An unspecified error occurred.")
                flask_logger.warning(f"API: Research for {project_id} marked as error-completed based on messages.")

        except json.JSONDecodeError as e_json:
            flask_logger.error(f"API: Error decoding state.json for {project_id}: {e_json}", exc_info=True)
            status_response["error_message"] = f"Error reading project state file (JSON invalid): {e_json}"
            status_response["completed"] = True 
        except Exception as e_state:
            flask_logger.error(f"API: Error reading state for {project_id}: {e_state}", exc_info=True)
            status_response["error_message"] = f"Server error reading project state: {e_state}"
            status_response["completed"] = True 
    else:
        status_response["current_node_message"] = "Project state file not yet created. Agent is likely initializing."
        flask_logger.info(f"API: State file not found for {project_id}, assuming initialization.")
        
    flask_logger.debug(f"API: Returning status for {project_id}: Completed={status_response['completed']}, CurrentNodeMsg='{status_response['current_node_message']}', MsgsCount={len(status_response['messages'])}")
    return jsonify(status_response)

@app.route('/research_projects/<project_id>/charts/<path:filename>') # Use <path:filename> for subdirs if any
def get_research_chart_route(project_id, filename):
    # No OPTIONS needed here as it's a simple GET for an image, usually not preflighted if from same effective origin via proxy
    flask_logger.debug(f"API: Received GET for chart: /research_projects/{project_id}/charts/{filename}")
    charts_dir = os.path.join(app.config['RESEARCH_PROJECTS_DIR'], project_id, "charts")
    # Basic security check
    if ".." in filename or filename.startswith("/"):
        flask_logger.warning(f"API: Invalid chart filename requested: {filename}")
        return "Invalid filename", 400
    
    # Check if file exists before sending
    if not os.path.exists(os.path.join(charts_dir, filename)):
        flask_logger.warning(f"API: Chart file not found: {os.path.join(charts_dir, filename)}")
        return "Chart not found", 404
        
    return send_from_directory(charts_dir, filename)

@app.route('/api/dashboard/stats', methods=['GET'])
def get_dashboard_stats():
    """Get statistics for the dashboard."""
    try:
        stats = database.get_project_stats()
        return jsonify(stats), 200
    except Exception as e:
        flask_logger.error(f"API: Error getting dashboard stats: {e}", exc_info=True)
        return jsonify({"error": f"Error getting dashboard stats: {str(e)}"}), 500

@app.route('/api/projects', methods=['GET'])
def get_projects():
    """Get a list of all research projects."""
    try:
        limit = request.args.get('limit', 50, type=int)
        offset = request.args.get('offset', 0, type=int)
        status = request.args.get('status')
        
        projects = database.get_all_projects(limit, offset, status)
        return jsonify({"projects": projects, "total": len(projects)}), 200
    except Exception as e:
        flask_logger.error(f"API: Error getting projects: {e}", exc_info=True)
        return jsonify({"error": f"Error getting projects: {str(e)}"}), 500

@app.route('/api/projects/<project_id>', methods=['GET'])
def get_project(project_id):
    """Get details for a specific project."""
    try:
        project = database.get_project(project_id)
        if not project:
            return jsonify({"error": "Project not found"}), 404
        return jsonify(project), 200
    except Exception as e:
        flask_logger.error(f"API: Error getting project {project_id}: {e}", exc_info=True)
        return jsonify({"error": f"Error getting project: {str(e)}"}), 500

@app.route('/api/projects/<project_id>', methods=['DELETE'])
def delete_project(project_id):
    """Delete a project."""
    try:
        success = database.delete_project(project_id)
        if not success:
            return jsonify({"error": "Failed to delete project"}), 500
        return jsonify({"message": "Project deleted successfully"}), 200
    except Exception as e:
        flask_logger.error(f"API: Error deleting project {project_id}: {e}", exc_info=True)
        return jsonify({"error": f"Error deleting project: {str(e)}"}), 500

@app.route('/api/settings', methods=['GET'])
def get_settings():
    """Get application settings."""
    try:
        # Get API key (masked for security)
        gemini_key = os.getenv("GEMINI_API_KEY", "")
        masked_key = ""
        if gemini_key:
            # Mask the API key, showing only first and last 4 characters
            if len(gemini_key) > 8:
                masked_key = gemini_key[:4] + "*" * (len(gemini_key) - 8) + gemini_key[-4:]
            else:
                masked_key = "****"
        
        # Get other settings from environment variables
        settings = {
            "api_key": masked_key,
            "api_key_set": bool(gemini_key),
            "api_model": os.getenv("API_MODEL", "gemini-1.5-flash"),
            "temperature": float(os.getenv("TEMPERATURE", "0.3")),
            "max_projects": int(os.getenv("MAX_PROJECTS", "50")),
            "storage_location": "local",  # Currently only local SQLite is supported
            "debug_mode": os.getenv("DEBUG", "False").lower() == "true"
        }
        
        return jsonify(settings), 200
    except Exception as e:
        flask_logger.error(f"API: Error getting settings: {e}", exc_info=True)
        return jsonify({"error": f"Error getting settings: {str(e)}"}), 500

@app.route('/api/settings', methods=['POST'])
def update_settings():
    """Update application settings."""
    try:
        if not request.is_json:
            return jsonify({"error": "Request must be JSON"}), 400
            
        data = request.get_json()
        
        # Update .env file with new settings
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
        
        # Read existing .env file
        env_vars = {}
        if os.path.exists(env_path):
            with open(env_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        key, value = line.split('=', 1)
                        env_vars[key] = value
        
        # Update with new values
        if 'api_key' in data and data['api_key'] and not data['api_key'].startswith('*'):
            env_vars['GEMINI_API_KEY'] = data['api_key']
            os.environ['GEMINI_API_KEY'] = data['api_key']
        
        if 'api_model' in data:
            env_vars['API_MODEL'] = data['api_model']
            os.environ['API_MODEL'] = data['api_model']
            
        if 'temperature' in data:
            env_vars['TEMPERATURE'] = str(data['temperature'])
            os.environ['TEMPERATURE'] = str(data['temperature'])
            
        if 'max_projects' in data:
            env_vars['MAX_PROJECTS'] = str(data['max_projects'])
            os.environ['MAX_PROJECTS'] = str(data['max_projects'])
            
        if 'debug_mode' in data:
            env_vars['DEBUG'] = str(data['debug_mode']).lower()
            os.environ['DEBUG'] = str(data['debug_mode']).lower()
        
        # Write back to .env file
        with open(env_path, 'w') as f:
            for key, value in env_vars.items():
                f.write(f"{key}={value}\n")
        
        return jsonify({"message": "Settings updated successfully"}), 200
    except Exception as e:
        flask_logger.error(f"API: Error updating settings: {e}", exc_info=True)
        return jsonify({"error": f"Error updating settings: {str(e)}"}), 500

if __name__ == '__main__':
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        flask_logger.critical("CRITICAL ERROR: GEMINI_API_KEY not found in .env file. The agent's LLM capabilities WILL BE DISABLED.")
    else:
        flask_logger.info("GEMINI_API_KEY found. LLM should be available to the agent.")
    flask_logger.info("Starting Flask research agent server on port 5001 with debug mode ON.")
    app.run(debug=True, host='0.0.0.0', port=5001, use_reloader=True) # use_reloader=True is default with debug=True