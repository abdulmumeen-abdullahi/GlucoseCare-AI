# api/index.py
from http.server import BaseHTTPRequestHandler
import json
import sys
import os

# Make sure Python can find GlucoseCare.py
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from GlucoseCare import app, ensure_state, memory

class handler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        body = self.rfile.read(content_length)
        data = json.loads(body)

        user_input = data.get("message", "")
        state = ensure_state({})
        state["input"] = user_input

        new_state = app.invoke(state, config={"configurable": {"thread_id": "web"}}, checkpointer=memory)
        response_text = new_state.get("output", "⚠️ No response")

        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"reply": response_text}).encode())
