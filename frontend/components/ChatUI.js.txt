import { useState } from "react";

export default function ChatUI() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");

  async function sendMessage() {
    if (!input.trim()) return;
    const userMessage = { role: "user", content: input };

    setMessages((prev) => [...prev, userMessage]);
    setInput("");

    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: input }),
    });

    const data = await res.json();
    setMessages((prev) => [...prev, { role: "assistant", content: data.response }]);
  }

  return (
    <div className="chatbox">
      <div className="messages">
        {messages.map((m, i) => (
          <div key={i} className={m.role}>
            <b>{m.role}:</b> {m.content}
          </div>
        ))}
      </div>
      <div className="input-area">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Type your question..."
        />
        <button onClick={sendMessage}>Send</button>
      </div>
    </div>
  );
}
