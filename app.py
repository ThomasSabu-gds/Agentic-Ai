import os
from flask import Flask, render_template, request, redirect, url_for, flash
from azure.data.tables import TableServiceClient
from multi_agent_autogen import run_pipeline
from dotenv import load_dotenv

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")  # change in prod

load_dotenv()
# -------------------------------
# Azure Table Storage
# -------------------------------

AZURE_CONN_STR = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "UseDevelopmentStorage=true"
)
TABLE_NAME = "AgentsTable"

service = TableServiceClient.from_connection_string(conn_str=AZURE_CONN_STR)
table_client = service.get_table_client(TABLE_NAME)

try:
    table_client.create_table()
except Exception:
    pass


# -------------------------------
# File validation
# -------------------------------

ALLOWED_EXTENSIONS = {"pdf", "png", "jpg", "jpeg", "tiff", "bmp"}
ALLOWED_DOC_TYPES = {"invoice", "receipt", "identity"}


def is_allowed_file(filename: str) -> bool:
    if "." not in filename:
        return False
    ext = filename.rsplit(".", 1)[1].lower()
    return ext in ALLOWED_EXTENSIONS


# -------------------------------
# Routes
# -------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    topic = ""
    result = None

    if request.method == "POST":
        topic = request.form.get("topic", "").strip()
        doc_type = request.form.get("doc_type", "").strip()
        uploaded_file = request.files.get("file")

        if not topic:
            flash("Please enter a task.")
            return redirect(url_for("index"))

        if doc_type not in ALLOWED_DOC_TYPES:
            flash("Please select a valid document type.")
            return redirect(url_for("index"))

        if not uploaded_file or not uploaded_file.filename:
            flash("Please upload a document.")
            return redirect(url_for("index"))

        if not is_allowed_file(uploaded_file.filename):
            flash("Unsupported file type. Please upload PDF or image files only.")
            return redirect(url_for("index"))

        file_bytes = uploaded_file.read()

        result = run_pipeline(
            task=topic,
            table_client=table_client,
            file_bytes=file_bytes,
            doc_type=doc_type
        )

    return render_template(
        "index.html",
        topic=topic,
        result=result
    )


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
            return redirect(url_for("agents_list"))
        except Exception as e:
            flash(f"Failed to create agent: {e}")
            return redirect(url_for("agents_list"))

    return render_template(
        "agents.html",
        agents=agents,
        allowed_models=allowed_models
    )


@app.route("/create_agent", methods=["GET", "POST"])
def create_agent():
    suggested_name = request.args.get("name", "")
    suggested_prompt = request.args.get("prompt", "")

    allowed_models = ["gpt-4.1-mini"]

    if request.method == "POST":
        agent_name = request.form.get("agent_name", "").strip()
        agent_prompt = request.form.get("agent_prompt", "").strip()
        model = request.form.get("model", "gpt-4.1-mini")

        if not agent_name or not agent_prompt:
            flash("Agent name and prompt are required.")
            return redirect(url_for("create_agent"))

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
            return redirect(url_for("index"))
        except Exception as e:
            flash(f"Failed to create agent: {e}")
            return redirect(url_for("create_agent"))

    return render_template(
        "create_agent.html",
        suggested_name=suggested_name,
        suggested_prompt=suggested_prompt,
        allowed_models=allowed_models
    )


# -------------------------------
# DB Helper
# -------------------------------

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


# -------------------------------
# Local Run
# -------------------------------

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
