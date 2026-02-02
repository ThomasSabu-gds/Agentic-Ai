import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, make_response

from multi_agent_autogen import AgenticAI, Settings
from dotenv import load_dotenv
from utils.utility import logger
from memory_store import Memory

# --------------------------------------------------
# APP INIT
# --------------------------------------------------
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")  # change in prod
settings = Settings()
GLOBAL_MEMORY = Memory()
client = AgenticAI(settings=settings)

# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------
ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp", "docx"}
ALLOWED_DOC_TYPES = {"invoice", "receipt", "identity", "summary"}


def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS

def is_ajax_request(req) -> bool:
    # fetch() commonly sends this header; we add it from JS
    return req.headers.get("X-Requested-With") == "XMLHttpRequest"


# --------------------------------------------------
# MAIN ROUTE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    topic = ""
    result = None

    if request.method == "POST":
        try:
            topic = request.form.get("topic", "").strip()
            # uploaded_files = request.files.getlist("file")
            uploaded_files = request.files.getlist("files")

            if not topic:
                msg = "Please enter a task."
                if is_ajax_request(request):
                    # return jsonify({"status": "error", "message": msg}), 400
                    
                    resp = make_response(jsonify({"status": "error", "message": msg}), 400)
                    sid = GLOBAL_MEMORY.get_or_create_session_id(request)
                    GLOBAL_MEMORY.attach_session_cookie(resp, sid)
                    return resp

                flash(msg)
                return redirect(url_for("index"))

            # Validate and prepare all uploaded files
            files_list = []
            for uploaded_file in uploaded_files:
                if uploaded_file and uploaded_file.filename:
                    if not is_allowed_file(uploaded_file.filename):
                        msg = f"Unsupported file type: {uploaded_file.filename}"
                        if is_ajax_request(request):
                            # return jsonify({"status": "error", "message": msg}), 400
                            resp = make_response(jsonify({"status": "error", "message": msg}), 400)
                            sid = GLOBAL_MEMORY.get_or_create_session_id(request)
                            GLOBAL_MEMORY.attach_session_cookie(resp, sid)
                            return resp
                        flash(msg)
                        return redirect(url_for("index"))
                    
                    file_bytes = uploaded_file.read()
                    files_list.append({
                        "bytes": file_bytes,
                        "filename": uploaded_file.filename
                    })

            session_id = GLOBAL_MEMORY.get_or_create_session_id(request)
            
            # Pass all files as a list to run_pipeline
            result = client.run_pipeline(
                task=topic,
                # table_client=table_client,
                session_id=session_id,
                memory_store=GLOBAL_MEMORY,
                files=files_list if files_list else None,
            )

            # Make sure result is dict-like
            if not isinstance(result, dict):
                result = {"status": "success", "output": result}

            # If AJAX, return JSON (no page reload)
            if is_ajax_request(request):
                # return jsonify(result)   
                resp = make_response(jsonify(result))
                GLOBAL_MEMORY.attach_session_cookie(resp, session_id)
                return resp


        except Exception as e:
            # Return proper error for AJAX or normal render for classic
            logger.error(f"Internal error: {str(e)}")
            err = {"status": "error", "message": f"Internal error: {str(e)}"}
            if is_ajax_request(request):
                # return jsonify(err), 500     
                resp = make_response(jsonify(err), 500)
                sid = GLOBAL_MEMORY.get_or_create_session_id(request)
                GLOBAL_MEMORY.attach_session_cookie(resp, sid)
                return resp

            result = err

    # return render_template("index.html", topic=topic, result=result )
    
    resp = make_response(render_template("index.html", topic=topic, result=result))
    sid = GLOBAL_MEMORY.get_or_create_session_id(request)
    GLOBAL_MEMORY.attach_session_cookie(resp, sid)
    return resp



# --------------------------------------------------
# AGENTS LIST
# --------------------------------------------------

@app.route("/agents", methods=["GET", "POST"])
def agents_list():
    allowed_models = ["gpt-4.1-mini"]
    agents = []

    try:
        agents = list(client.table_client.query_entities("PartitionKey eq 'agents'"))
    except Exception:
        logger.error("Unable to access agents from the table")
        pass

    if request.method == "POST":
        agent_name = request.form.get("agent_name", "").strip()
        agent_prompt = request.form.get("agent_prompt", "").strip()
        model = request.form.get("model", "gpt-4.1-mini")

        if not agent_name or not agent_prompt:
            flash("Agent name and prompt are required.")
            return redirect(url_for("agents_list"))

        row_key = agent_name.replace(" ", "")

        try:
            save_agent_to_db(
                client.table_client,
                row_key,
                agent_prompt,
                model,
                agent_type="llm"
            )
            flash(f"Agent '{agent_name}' created successfully.")
            logger.info(f"Agent {agent_name} created successfully")
        except Exception as e:
            logger.error("Failed to create agent")
            flash(f"Failed to create agent: {e}")

        return redirect(url_for("agents_list"))

    return render_template(
        "agents.html",
        agents=agents,
        allowed_models=allowed_models
    )


# --------------------------------------------------
# DB HELPER
# --------------------------------------------------

def save_agent_to_db(
    table_client,
    row_key,
    prompt,
    model="gpt-4.1-mini",
    agent_type="llm"
):
    entity = {
        "PartitionKey": "agents",
        "RowKey": row_key,
        "prompt": prompt,
        "model": model,
        "agent_type": agent_type,
        "base_url": os.environ.get("AZURE_OPENAI_BASE_URL", ""),
        "api_version": os.environ.get("DEFAULT_API_VERSION", ""),
        "api_key": ""
    }
    table_client.upsert_entity(entity)

# --------------------------------------------------
# LOCAL RUN
# --------------------------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))



