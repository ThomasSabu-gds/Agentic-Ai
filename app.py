import os
from flask import Flask, render_template, request, redirect, url_for, flash
from azure.data.tables import TableServiceClient
from multi_agent_autogen import run_pipeline

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "super-secret-key")  # change in prod


# Table storage connection
AZURE_CONN_STR = os.environ.get(
    "AZURE_STORAGE_CONNECTION_STRING",
    "UseDevelopmentStorage=true"
)
TABLE_NAME = "AgentsTable"

# Initialize table client
service = TableServiceClient.from_connection_string(conn_str=AZURE_CONN_STR)
table_client = service.get_table_client(TABLE_NAME)

# Ensure table exists
try:
    table_client.create_table()
except Exception:
    pass


# Routes

@app.route("/", methods=["GET", "POST"])
def index():
    topic = ""
    result = None

    if request.method == "POST":
        topic = request.form.get("topic", "").strip()

        if not topic:
            flash("Please enter a task.")
            return redirect(url_for("index"))

        result = run_pipeline(topic, table_client)

    return render_template(
        "index.html",
        topic=topic,
        result=result
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

        # Normalize RowKey
        row_key = agent_name.replace(" ", "")

        try:
            save_agent_to_db(
                table_client,
                row_key,
                agent_prompt,
                model,
                agent_type="llm"   # ✅ DEFAULT FOR USER AGENTS
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


# DB Helper

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
        "agent_type": agent_type,  # ✅ NEW FIELD
        "base_url": os.environ.get("AZURE_OPENAI_BASE_URL", ""),
        "api_version": os.environ.get("DEFAULT_API_VERSION", ""),
        "api_key": ""
    }
    table_client.upsert_entity(entity)


# Local Run (Azure uses gunicorn)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)))
