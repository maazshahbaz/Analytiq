# agents/pandas_agent/helper.py
import openai, os, time
import streamlit as st
openai.api_key = st.secrets["OPENAI_API_KEY"]

def create_assistant(name="Excel Analyst"):
    """One‑time creation.  Returns assistant_id."""
    assistant = openai.beta.assistants.create(
        name=name,
        model="gpt‑4o‑mini",          # or "o1"
        tools=[{"type": "code_interpreter"}],
        instructions=(
            "You are a seasoned data‑analytics assistant. "
            "When given a spreadsheet or CSV you must:\n"
            "• Load *all* sheets with pandas\n"
            "• Give a concise natural‑language overview\n"
            "• Answer follow‑up questions with clear explanations\n"
            "• Produce plots when they help. Use bar/line/scatter by default\n"
            "• Never modify the original file\n"
        )
    )
    return assistant.id

# cache this on disk or in memory; create once
ASSISTANT_ID = create_assistant()

def ask_excel(question: str, file_path: str, thread_id: str | None = None):
    """
    • Uploads/attaches the file on first call (gets file_id)
    • Sends the user question
    • Waits for run completion
    • Returns the text answer + list of image URLs (if any)
    """
    # 1. upload file only once per thread
    if thread_id is None:
        thread = openai.beta.threads.create()
        thread_id = thread.id
        file_obj = openai.files.create(
            file=open(file_path, "rb"), purpose="assistants"
        )
        file_id = file_obj.id
        # Attach the file in an initial message so code_interpreter can read it
        openai.beta.threads.messages.create(
            thread_id, role="user",
            content="Here is the dataset.",
            file_ids=[file_id]
        )

    # 2. ask the question
    openai.beta.threads.messages.create(
        thread_id, role="user", content=question
    )
    run = openai.beta.threads.runs.create(
        thread_id, assistant_id=ASSISTANT_ID
    )

    # 3. simple polling
    while True:
        run = openai.beta.threads.runs.retrieve(thread_id, run.id)
        if run.status in {"completed", "failed"}:
            break
        time.sleep(1)

    if run.status == "failed":
        raise RuntimeError("Assistant run failed")

    # 4. collect outputs
    msgs = openai.beta.threads.messages.list(thread_id, order="asc")
    answer = msgs.data[-1].content[0].text.value

    # any images produced by code_interpreter will be in `run.steps`
    images = []
    for step in run.steps.data:
        for out in step.outputs:
            if out.type == "image":
                images.append(out.image.file_id)

    return answer, images, thread_id
