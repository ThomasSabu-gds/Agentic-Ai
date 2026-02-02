import os, json
from datetime import datetime
from typing import Dict, Optional, Any
from utils import utility
from copy import deepcopy
from dataclasses import dataclass
from dotenv import load_dotenv
from autogen.agentchat import AssistantAgent
from azure.ai.formrecognizer import DocumentAnalysisClient
from azure.core.credentials import AzureKeyCredential
from azure.data.tables import TableServiceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest
from utils.utility import logger
 

load_dotenv()


@dataclass
class Settings:
    azure_endpoint: str   = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    azure_api_key: str    = os.getenv("AZURE_OPENAI_API_KEY", "")
    azure_api_version: str= os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
    AVAILABLE_MODELS = {
        "gpt-4.1-mini": {
            "model": "gpt-4.1-mini",
            "api_key": os.environ.get("AZURE_OPENAI_API_KEY"),
            "api_type": "azure",
            "base_url": os.environ.get(
                "AZURE_OPENAI_BASE_URL",
                "https://model-azure.openai.azure.com/"
            ),
            "api_version": os.environ.get(
                "DEFAULT_API_VERSION",
                "2025-01-01-preview"
            ),
            "temperature": 0.4,
            "max_tokens": 1000,
        }
    }
    AZURE_CONN_STR = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "UseDevelopmentStorage=true")
    TABLE_NAME = "AgentsTable"
    AZURE_DI_ENDPOINT = os.environ.get("AZURE_DI_ENDPOINT")
    AZURE_DI_KEY = os.environ.get("AZURE_DI_KEY")



class AgenticAI:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.DOCUMENT_TYPES = {
            "invoice": {
                "prebuilt-invoice":  {
                    "fields": ["VendorName", "VendorAddress", "VendorAddressRecipient", "CustomerName", "CustomerId", "CustomerAddress", "CustomerAddressRecipient", "InvoiceId", "InvoiceDate", "InvoiceTotal", "DueDate", "PurchaseOrder", "BillingAddress", "BillingAddressRecipient", "ShippingAddress", "ShippingAddressRecipient","SubTotal", "TotalTax", "PreviousUnpaidBalance", "AmountDue", "ServiceStartDate", "ServiceEndDate", "ServiceAddress", "ServiceAddressRecipient", "RemittanceAddress", "RemittanceAddressRecipient", ],
                    "items": ["Description", "Quantity", "Unit", "UnitPrice", "ProductCode", "Date", "Tax", "Amount", ],
                    "mask": ["VendorName", "VendorAddress", "CustomerName", "CustomerAddress", "VendorAddressRecipient", "CustomerAddressRecipient"]
                    }},
            "receipt": {
                "prebuilt-receipt": {
                    "fields": ["MerchantName", "TransactionDate", "Subtotal", "TotalTax", "Tip", "Total"],
                    "items": ["Description", "Quantity", "Price", "TotalPrice"],
                    "mask": ["MerchantName"]
                    }},
            "identity": {
                "prebuilt-idDocument": {
                    "fields": ["FirstName", "LastName", "DateOfBirth", "DocumentNumber", "DateOfExpiration", "Address", "Sex" , "CountryRegion", "Region"],
                    "items" : [],
                    "mask": []
                    }},
            "general_document": {
                "prebuilt-document": {
                    "fields": [],
                    "items": [],
                    "mask": []
            }}}
        self.table_client = self.table_init()
        self.agents_meta = self.load_agents_from_db(self.table_client)
        self.supervisor = None


    def table_init(self):
        service = TableServiceClient.from_connection_string(self.settings.AZURE_CONN_STR)
        table_client = service.get_table_client(self.settings.TABLE_NAME)
        try:
            table_client.create_table_if_not_exists()
        except Exception:
            logger.error("Error in setting table client")
        finally:
            return table_client


    def build_llm_config(self, model_key: str) -> dict:
        info = self.settings.AVAILABLE_MODELS[model_key]
        return {
            "config_list": [
                {
                    "model": info["model"],
                    "api_key": info["api_key"],
                    "api_type": info["api_type"],
                    "base_url": info["base_url"],
                    "api_version": info["api_version"],
                }
            ],
            "temperature": info["temperature"],
            "max_tokens": info["max_tokens"],
        }
    
    def load_agents_from_db(self, table_client) -> Dict[str, dict]:
        agents = {}
    
        for ent in table_client.list_entities():
            if ent.get("PartitionKey") != "agents":
                continue
    
            name = ent["RowKey"].strip()
            if not name.isidentifier():
                continue
    
            agents[name] = {
                "name": name,
                "prompt": ent.get("prompt", ""),
                "model": ent.get("model", "gpt-4.1-mini"),
                "agent_type": ent.get("agent_type", "llm"),  # llm | service
            }
        
        if "Supervisor" not in agents:
            logger.error("Supervisor agent missing in the database")
            return {
                "status": "error",
                "message": "Supervisor agent missing in database."
            }
        
        # agents_meta = {'DocumentIntelligence': {'name': 'DocumentIntelligence', 'prompt': 'You extract ALL readable text from uploaded documents such as PDFs and images.\nYou do not summarize, explain, or modify the text.\nYou must be selected when the task mentions documents, files, PDFs, images, or extracting text.\nYour output is only the extracted text.\n', 'model': 'gpt-4.1-mini', 'agent_type': 'service'}, 
        #                'QuizMaker':
        #                         {'name': 'QuizMaker', 'prompt': 'You are a quiz generation agent.\n\nYour role is to create high-quality quiz questions to test understanding of a topic or text.\n\nYou can handle requests such as:\n- generating quizzes on a topic\n- creating practice questions for learning or revision\n- assessing knowledge using multiple questions\n- turning explanations or summaries into quizzes\n\nGuidelines:\n- Generate exactly 5 questions unless the user explicitly asks otherwise\n- Each question must be clear and unambiguous\n- Each answer must be ONE WORD\n- Cover different aspects of the topic when possible\n\nOutput format (MANDATORY):\nQ1: &lt;question&gt;\nA1: &lt;one-word-answer&gt;\n...\nQ5: &lt;question&gt;\nA5: &lt;one-word-answer&gt;', 'model': 'gpt-4.1-mini', 'agent_type': 'llm'}, 
        #                'Supervisor': 
        #                         {'name': 'Supervisor', 
        #                         'prompt': 'You are a supervisor agent. You will be given: 1. A user task 2. A list of available agents and their roles 3. Information on whether a document file is uploaded Your job: - Select ONLY ONE agent that best fits the user task - Select the DocumentIntelligence agent ONLY when: - The task mentions documents, files, PDFs, images, or extracting text - AND a document file is uploaded - If the task requires a document but NO file is uploaded, return NONE - If the task does not require a document, select an appropriate LLM agent - If no agent is suitable, return NONE Output rules (MANDATORY): - Output ONLY the agent name OR NONE - Do NOT explain anything - Do NOT generate task content - Do NOT talk to the user Output format (MANDATORY): <AgentName> or NONE', 
        #                         'model': 'gpt-4.1-mini', 
        #                         'agent_type': 'llm'},
        #                'DocQA':
        #                         {'name': 'DocQA',
        #                         'prompt': "You must ALWAYS operate only on the content inside <document>...</document>. Never use outside knowledge. FIRST determine the user's intent, then respond accordingly. 1) EXTRACT/READ/SHOW/DISPLAY → Return FULL document text verbatim (no bullets, no markdown). 2) OVERVIEW/SUMMARY → 2–4 sentences, no bullets. 3) SPECIFIC QUESTION → Answer concisely; include: Source: "<exact line from document>" 4) If truly absent → reply exactly: Not found in document Never reformat extracted text. Never invent or infer.'.",
        #                         'model': 'gpt-4.1-mini',
        #                         'agent_type': 'llm'
        #                 }}

        logger.info("Agents fetched succesfully")
        return agents
    
    def build_agent_catalog(self, file_bytes: Optional[bytes]) -> str:
        lines = []
    
        for name, meta in self.agents_meta.items():
            if name == "Supervisor":
                continue
    
            if meta.get("agent_type") == "service" and not file_bytes:
                continue
    
            role = meta["prompt"].replace("\n", " ").strip()
            lines.append(f"- {name}: {role}")
    
        return "\n".join(lines)
    
    def fetch_results(self, results, ext_type:str, model_id: str) -> Dict[str, Any]:
        if not results:
            logger.error( "No results provided.")
            return {
                "status": "error",
                "message": "No results provided."
            }
        # print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv", ext_type, model_id)
        doc_type = self.DOCUMENT_TYPES.get(ext_type.lower())        
        out = {}

        # if doc_type not in DOC_TYPE_FIELDS.keys():
        if not doc_type:
            logger.error("Unsupported document type for result fetching.")
            return {
                "status": "error",
                "message": "Unsupported document type for result fetching."
            }
        # elif doc_type == "prebuilt-document":
        elif "prebuilt-document" in doc_type:
            for kv_pair in results.key_value_pairs:
                key = kv_pair.key.content.strip() if kv_pair.key and kv_pair.key.content else None
                value = kv_pair.value.content.strip() if kv_pair.value and kv_pair.value.content else None
                
                if key:  
                    out[key] = value
            
            return out

        for idx, document in enumerate(results.documents):
            # for field in self.DOCUMENT_TYPES.get(ext_type, {}).get(doc_type, {}).get("fields", []):
            for field in doc_type.get(model_id, {}).get("fields", []):
                if not document.fields.get(field) or document.fields.get(field).content.lower() in {"n/a", "na", "none", "null", ""}:
                            continue
                out[field] = document.fields.get(field, None).content.replace("\n", "") if document.fields.get(field) else None
        if document.fields.get("Items"):
            # print("Processing line items...", document.fields.get("Items"))
            for idx, itemval in enumerate(document.fields.get("Items").value):
                # print("Item value:", itemval.value)
                for keyys in itemval.value.keys():
                    # if keyys in self.DOCUMENT_TYPES.get(ext_type, {}).get(doc_type, {}).get("items",[]):
                    if keyys in doc_type.get(model_id, {}).get("items",[]):
                        # print("Processing item:", itemval.value)
                        if not itemval.value.get(keyys) or itemval.value.get(keyys).content.lower() in {"n/a", "na", "none", "null", ""}:
                            continue
                        out[f"Item_{idx+1}_{keyys}"] = itemval.value.get(keyys).content.replace("\n", "")
            
        return out
    

    def run_form_recognizer(self, file_bytes: bytes, ext_type: str, model_id:str,  filename: str ) -> str:

        if not self.settings.AZURE_DI_ENDPOINT or not self.settings.AZURE_DI_KEY:
            logger.info("Document Intelligence credentials not configured.")
            return "Document Intelligence credentials not configured."
    
        client = DocumentAnalysisClient(
            endpoint=self.settings.AZURE_DI_ENDPOINT,
            credential=AzureKeyCredential(self.settings.AZURE_DI_KEY)
        )

        # model_id = next(iter(self.DOCUMENT_TYPES[ext_type.lower()]))
        # model_id = self.DOCUMENT_TYPES.get(ext_type)

        poller = client.begin_analyze_document(model_id=model_id, document=file_bytes)
        result = poller.result()
        raw_full = result.to_dict()

        raw_slim = utility._strip_noise(deepcopy(raw_full))

        flat_values = {}
        if result.documents:
            doc = result.documents[0]
            if doc.fields:
                for fname, f in doc.fields.items():
                    flat_values[fname] = utility._field_value_to_python(f)

        # flat_text = "\n".join(f"{k}: {v}" for k, v in flat_values.items())
        # print("Flat field values:", flat_values)
        # print("Slimmed raw result:", raw_slim)
        
        out = self.fetch_results(result, ext_type, model_id)
        # print("Fetched results:", out)
        logger.info("Fetched documents")
        return {
            # "raw_full": raw_full,
            "raw_slim": out,
            "flat_values": raw_slim,
        }
    
    
    def mask_document(self, ext_type, model_id, raw_dict):
        masked = {}
        pii = set(self.DOCUMENT_TYPES.get(ext_type,{}).get(model_id, {}).get("mask", []))
    
        def mask_value(value: str):
            if not value:
                return value
            if len(value) <= 4:
                return "***"
            return value[:2] + "***" + value[-2:]

        for k, v in raw_dict.items():
            if k in pii:
                masked[k] = mask_value(v)
            else:
                masked[k] = v
        return masked

    
    def run_pipeline( self, task: str, session_id, memory_store, files: Optional[list] = None, sticky_document: bool = True) -> dict:

        if not task or not task.strip():
            logger.error("No task provided........please provide a task")
            return {"status": "error", "message": "Please provide a task."}

        task_l = task.lower().strip()
        files = files or []  #normalize to list (avoid len(None))
        has_file_upload = bool(files and files[0].get("bytes"))
        has_session_doc = bool(session_id and memory_store.has_doc(session_id))

        logger.info(f"session-id {session_id}, \nhas_file uploaded, {has_file_upload}, \nhas session doc, {has_session_doc}")

        
        new_fingerprints, new_hashes = [], []
        if has_file_upload:
            for f in files:
                fp = utility._file_fingerprint(f)
                if not fp:
                    continue
                new_fingerprints.append(fp)
                new_hashes.append(fp["sha256"])
        new_hash_set = set(new_hashes)

        #prior meta
        prior_meta = memory_store.get_meta(session_id) if (session_id and has_session_doc) else {}
        prior_hash_set = set(prior_meta.get("file_hashes", []))


        #clr commands
        CLEAR_COMMANDS = ("clear document", "forget document", "reset document", "remove document")
        if any(c in task_l for c in CLEAR_COMMANDS):
            if session_id:
                memory_store.clear(session_id)
                logger.info("Document context cleared")
                return {"status": "success", "agent": "System", "output": "Document context cleared."}
            else:
                logger.info("No session_id provided; cannot clear document context.")
                return {"status": "error", "message": "No session_id provided; cannot clear document context."}

        #xtract intent detection
        EXTRACT_INTENTS = {
            "extract", "extract the file", "extract file",
            "read", "read document", "display document",
            "show", "show contents", "what are in the doc", "what is in the doc",
            "what does this contain"
        }
        is_extract_intent = task_l in EXTRACT_INTENTS

        
        if sticky_document and has_session_doc and has_file_upload and new_hash_set and (new_hash_set == prior_hash_set):
            logger.info("Identical upload detected for this session; skipping re-extraction and using the stored document.")
            files = []
            has_file_upload = False


        #no new file---condition
        if has_session_doc and not has_file_upload:
            document_blob = memory_store.get_doc(session_id, consume_if_single_use=(not sticky_document))
            if is_extract_intent:
                logger.info("No new file-----fallback to memory")
                return {
                    "status": "success",
                    "agent": "DocExtract",
                    "output": document_blob
                }

            qa_agent = AssistantAgent(
                name="DocQA",
                system_message="""

                                """,
                llm_config=self.build_llm_config("gpt-4.1-mini"),
            )

            answer = qa_agent.generate_reply(messages=[{
                                        "role": "user",
                                        "content": f"""
                                        <document>
                                        {document_blob}
                                        </document>

                                        User task:
                                        {task}
                                        """.strip()
            }])

            answer_text = answer.get("content", answer) if isinstance(answer, dict) else str(answer)
            logger.info("answers generatedd")
            return {"status": "success", "agent": "DocQA", "output": answer_text}

        #if the task mentions documents but no file uploaded and no session doc >>>>> early message
        if not has_file_upload and not has_session_doc:
            if any(word in task_l for word in ["invoice", "receipt", "document", "extract", "summarize", "tax"]):
                logger.info("noo document found")
                return {"status": "success", "agent": "System", "output": "No document found."}


        if has_file_upload:
            MULTI_LANGS = ["en", "ar"]
            joined_text = []
            joined_unmasked = []

            for f in files:
                file_bytes = f.get("bytes")
                filename = f.get("filename", "uploaded.bin")
                if not file_bytes:
                    continue

                #ocr
                text_ext = utility.extract_text_from_file(file_bytes, filename, langs=MULTI_LANGS)
                ext_type = utility.classify_text(text_ext)
                if ext_type.lower() not in self.DOCUMENT_TYPES.keys():
                    joined_text.append(text_ext)
                    continue

                model_id = next(iter(self.DOCUMENT_TYPES[ext_type.lower()]))
                # fr_out = self.run_form_recognizer(file_bytes, model_id, filename)
                fr_out = self.run_form_recognizer(file_bytes, ext_type, model_id, filename)
                logger.info(f"file---{filename}---ext_type---{ext_type}---model_id---{model_id}")

                extracted_text = "\n".join(f"{k}: {v}" for k, v in fr_out["raw_slim"].items())
                # print("vvvvvvvvvvvvvvvvvvvvvvvvvvv",extracted_text)
                masked_dict = self.mask_document(ext_type, model_id, fr_out["raw_slim"])
                masked_text = "\n".join(f"{k}: {v}" for k, v in masked_dict.items())

                joined_text.append(extracted_text)
                joined_unmasked.append(masked_text)

            document_blob = "\n\n--- PAGE BREAK ---\n\n".join(joined_text)
            document_blob_masked = "\n\n--- PAGE BREAK ---\n\n".join(joined_unmasked)
            print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;", document_blob_masked)
            print(";;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;", document_blob)

            if session_id:
                memory_store.set_doc(session_id, document_blob, document_blob_masked, sticky=sticky_document)
                
                meta_to_store = {
                    "file_hashes": sorted(list(set(new_hashes))),  
                    "files": new_fingerprints,                    
                    "updated_at": datetime.now().isoformat() + "Z",
                }
                memory_store.set_meta(session_id, meta_to_store)

            
            raw_blob = memory_store.get_doc(session_id, masked=False, consume_if_single_use=(not sticky_document))
            masked_blob = memory_store.get_doc(session_id, masked=True, consume_if_single_use=False)


            #for extract case
            if is_extract_intent:
                logger.info("No new file-----fallback to memory")
                return {"status": "success", "agent": "DocExtract", "output": raw_blob}
            
            docqa_meta = self.agents_meta.get("DocQA", None)
            if not docqa_meta:
                logger.error("couldnt find DocQA in agents meta")
                return {
                    "status": "failed",
                    "message": "couldnt find DocQA in agents meta"
                }

            #new doc
            qa_agent = AssistantAgent(
                name=docqa_meta.get('name'),
                # system_message="You are a document question answering assistant. Answer ONLY from the document. If absent: Not found in document.",
                system_message=docqa_meta.get('prompt'),
                llm_config=self.build_llm_config("gpt-4.1-mini"),
            )

            logger.debug("LLM doc preview (masked) = %s", masked_blob[:200])
            answer = qa_agent.generate_reply(messages=[{
                "role": "user",
                "content": f"<document>\n{masked_blob}\n</document>\n\nUser task:\n{task}"
            }])
            answer_text = answer.get("content", answer) if isinstance(answer, dict) else str(answer)
            logger.info("output generated with DocQA")
            return {"status": "success", "agent": "DocQA", "output": answer_text}

        #######supervisor
        if not self.supervisor:
            supervisor_meta = self.agents_meta["Supervisor"]

            supervisor = AssistantAgent(
                name="Supervisor",
                system_message=supervisor_meta["prompt"],
                llm_config=self.build_llm_config(supervisor_meta["model"]),
            )

            agent_catalog = self.build_agent_catalog(files[0]["bytes"] if files else None)
            # print("Agent Catalog:\n", agent_catalog)

        supervisor_input = f"""
                            USER TASK:
                            {task}

                            FILE_UPLOADED:
                            {"YES" if (files and len(files) > 0 and files[0].get("bytes")) else "NO"}

                            AVAILABLE AGENTS:
                            {agent_catalog}
                            """.strip()
    

        selected_agent = supervisor.generate_reply(
            messages=[{"role": "user", "content": supervisor_input}]
        ).strip()


        if selected_agent == "NONE":
            return {
                "status": "no_suitable_agent",
                "message": "No suitable agent found for this task."
            }

        if selected_agent not in self.agents_meta:
            return {
                "status": "error",
                "message": "Supervisor selected an unknown agent.",
                "raw": selected_agent
            }

        agent_meta = self.agents_meta[selected_agent]
        agent = AssistantAgent(
            name=agent_meta["name"],
            system_message=agent_meta["prompt"],
            llm_config=self.build_llm_config(agent_meta["model"]),
        )

        output = agent.generate_reply(
            messages=[{"role": "user", "content": task}]
        )

        return {
            "status": "success",
            "agent": selected_agent,
            "output": utility.format_output(output)
        }

