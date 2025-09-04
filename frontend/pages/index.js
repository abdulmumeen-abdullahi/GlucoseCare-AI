import ChatUI from "../components/ChatUI";

export default function Home() {
  return (
    <div className="container">
      <h1>GlucoseCare AI</h1>
      <p>Ask me about diabetes risk, symptoms, or lifestyle advice.</p>
      <ChatUI />
    </div>
  );
}

