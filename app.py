import os
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from azure.data.tables import TableServiceClient
from multi_agent_autogen import run_pipeline
from dotenv import load_dotenv

# --------------------------------------------------
# APP INIT
# --------------------------------------------------

load_dotenv()

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")

# --------------------------------------------------
# AZURE TABLE STORAGE
# --------------------------------------------------

AZURE_CONN_STR = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "UseDevelopmentStorage=true"
)
TABLE_NAME = "AgentsTable"

service = TableServiceClient.from_connection_string(AZURE_CONN_STR)
table_client = service.get_table_client(TABLE_NAME)

try:
    table_client.create_table()
except Exception:
    pass

# --------------------------------------------------
# FILE VALIDATION
# --------------------------------------------------

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp", "docx"}

def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


def is_ajax_request(req) -> bool:
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
            uploaded_files = request.files.getlist("file")

            if topic.lower() == "no":
                return jsonify({
                    "status": "success",
                    "agent": "System",
                    "output": "You have ended the chat."
                })

            if not topic:
                msg = "Please enter a task."
                if is_ajax_request(request):
                    return jsonify({"status": "error", "message": msg}), 400
                flash(msg)
                return redirect(url_for("index"))

            files_data = []

            for f in uploaded_files:
                if f and f.filename:
                    if not is_allowed_file(f.filename):
                        msg = "Unsupported file type."
                        if is_ajax_request(request):
                            return jsonify({"status": "error", "message": msg}), 400
                        flash(msg)
                        return redirect(url_for("index"))

                    files_data.append({
                        "filename": f.filename,
                        "bytes": f.read()
                    })


            all_outputs = []

            for file in files_data:
                result = run_pipeline(
                    task=topic,
                    table_client=table_client,
                    file_bytes=file["bytes"],
                    filename=file["filename"]
                )

                all_outputs.append({
                    "filename": file["filename"],
                    "text": result.get("output", ""),
                    "needs_confirmation": result.get("needs_confirmation", False)
                })

            # ✅ If only one file → behave like old system
            if len(all_outputs) == 1:
                final_result = {
                    "status": "success",
                    "agent": "System",
                    "output": all_outputs[0]["text"],
                    "needs_confirmation": all_outputs[0]["needs_confirmation"]
                }
            else:
                # ✅ Multiple files → show per file
                combined = ""
                needs_confirmation = False

                for o in all_outputs:
                    combined += f"\n\nIn file {o['filename']}:\n{o['text']}\n"
                    if o["needs_confirmation"]:
                        needs_confirmation = True

                final_result = {
                    "status": "success",
                    "agent": "System",
                    "output": combined.strip(),
                    "needs_confirmation": needs_confirmation
                }


            if is_ajax_request(request):
                return jsonify(final_result)


        except Exception as e:
            err = {"status": "error", "message": f"Internal error: {str(e)}"}
            if is_ajax_request(request):
                return jsonify(err), 500
            result = err

    return render_template(
        "index.html",
        topic=topic,
        result=result
    )

# --------------------------------------------------
# AGENTS LIST
# --------------------------------------------------

@app.route("/agents", methods=["GET", "POST"])
def agents_list():
    allowed_models = ["gpt-4.1-mini"]
    agents = []

    try:
        agents = list(table_client.query_entities("PartitionKey eq 'agents'"))
    except Exception:
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
                table_client,
                row_key,
                agent_prompt,
                model,
                agent_type="llm"
            )
            flash(f"Agent '{agent_name}' created successfully.")
        except Exception as e:
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


