# 🧑‍🍳 WAIter: Voice Chatbot for Restaurants

WAIter is an interactive voice-enabled chatbot designed for restaurant scenarios. It simulates a friendly and helpful waiter who can answer customers' spoken questions in real time — using speech-to-text, LLMs for conversation, and TTS for spoken responses.

## 🛠️ Technical Stack

- **Voice Processing**:
  - 🎙️ [FastRTC](https://fastrtc.org) for real-time audio streaming
  - 🔊 [ElevenLabs](https://elevenlabs.io/) for expressive Text-to-Speech
  - 🗣️ [Whisper](https://openai.com/research/whisper) (via Groq) for speech recognition

- **Language AI**:
  - 💬 [LangChain](https://www.langchain.com/) for LLM orchestration
  - 🔄 [LangGraph](https://github.com/langchain-ai/langgraph) for agent workflows
  - 🧠 [Gemini 2.5](https://deepmind.google/technologies/gemini/) (via OpenRouter) for natural language processing

- **Data Management**:
  - 🔍 Vector embeddings (HuggingFace) for menu information retrieval
  - 💾 Menu data stored in Markdown format

- **Interface**:
  - 🧪 [Gradio](https://www.gradio.app/) for the interactive web interface

## 🚀 Features

- **Real-time voice interaction** with a virtual waiter assistant
- **Menu information retrieval** with context-aware answers about dishes, prices and options
- **Order processing** with ability to send orders directly to kitchen systems
- **Natural conversational flow** using LangGraph for complex agent orchestration
- **Knowledge-base integration** with RAG (Retrieval Augmented Generation)
- **Multi-turn conversation memory** for context-aware responses

## 🚀 Getting Started

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt` (ensure you have Python 3.10+)
3. Create a `.env` file with your API keys:
   ```
   OPENROUTER_API_KEY=your_openrouter_key
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   ELEVENLABS_API_KEY=your_elevenlabs_key
   HELICONE_API_KEY=your_helicone_key_optional
   SUPABASE_URL=your_supabase_url
   SUPABASE_KEY=your_supabase_key
   ```
4. Run the application: `python app.py`
5. Open the Gradio interface in your browser

## 🗂️ Project Structure

- `app.py` - Main application entry point with Gradio interface
- `agent.py` - LangGraph agent implementation for restaurant assistant
- `tools.py` - Custom tools for menu lookup, and kitchen orders
- `functions.py` - Helper functions for message handling
- `supabase_client.py` - Client for database operations (if applicable)
- `data/carta.md` - Restaurant menu data in Markdown format
- `utils/` - Utility modules for logging and data classes

## 🧠 How It Works

1. **Speech Recognition**: FastRTC captures audio from the user's microphone and sends it to Groq's Whisper API for transcription
2. **Natural Language Understanding**: Transcribed text is passed to the LangGraph agent
3. **RAG Processing**: The agent uses vector search to find relevant menu information
4. **Response Generation**: Gemini 2.5 generates a contextual, helpful response
5. **Text-to-Speech**: ElevenLabs converts the text response to natural-sounding speech
6. **Web Interface**: Gradio renders the conversation and plays the audio response

## 📝 License

[MIT License](LICENSE)


