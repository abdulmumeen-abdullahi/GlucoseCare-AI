from fastapi import FastAPI
from pydantic import BaseModel
from GlucoseCare import app as agent_app, ensure_state, memory

app = FastAPI()  # not app_api anymore

class UserMessage(BaseModel):
    message: str
    thread_id: str | None = None

@app.post("/chat")
def chat(payload: UserMessage):
    state = ensure_state({})
    state["input"] = payload.message
    thread_id = payload.thread_id or "default"

    new_state = agent_app.invoke(
        state,
        config={"configurable": {"thread_id": thread_id}},
        checkpointer=memory
    )
    return {"response": new_state.get("output", "No response.")}

