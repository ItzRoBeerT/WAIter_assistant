import gradio as gr
from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv
from groq import Groq, AsyncClient
from fastrtc import WebRTC, ReplyOnPause, audio_to_bytes, get_tts_model, KokoroTTSOptions, AdditionalOutputs
import numpy as np
from openai import OpenAI

load_dotenv()

# constantes
RESTAURANT = "Bar paco"
SYSTEM_PROMPT = {"role": "system", "content": "Responde como si fueras un camarero amigable, servicial y divertido."}

# Clients
#groq_client = Groq()
groq_client = AsyncClient()

# LLM
llm = ChatOpenAI(
  openai_api_key=getenv("OPENROUTER_API_KEY"),
  openai_api_base=getenv("OPENROUTER_BASE_URL"),
  model_name="google/gemini-2.0-flash-001",
  model_kwargs={
    "extra_headers":{
        "Helicone-Auth": f"Bearer "+getenv("HELICONE_API_KEY")
      }
  },
)

model_tts = get_tts_model("kokoro")

# region FUNCIONS
async def response(audio: tuple[int, np.ndarray], history = []):
    """Return audio frames with complete response"""

    if history is None:
        history = []

    # Get transcript from audio
    transcript = await groq_client.audio.transcriptions.create(
        file=("audio-file.mp3", audio_to_bytes(audio)),
        model="whisper-large-v3-turbo",
        response_format="verbose_json",)

    user_text = transcript.text
    print(f"Transcripción: {user_text}")

    # Actualizar inmediatamente el chatbot con el mensaje del usuario
    user_message = {"role": "user", "content": user_text}
    updated_history = history + [user_message]
    yield AdditionalOutputs(updated_history)

    # Create prompt for the LLM
    final_prompt = (
        f"Eres un camarero experto para el restaurante {RESTAURANT}. "
        "Debes responder a las preguntas de los clientes y ayudarles a tener una experiencia agradable. "
        "Si no sabes la respuesta, puedes decir que no sabes o pedir ayuda a un compañero. "
        "Recuerda que la amabilidad y la paciencia son fundamentales. "
        "Si te pregunta por la carta, responde que la carta está en la mesa. "
        f"A continuación, un cliente te hace una pregunta: {user_text}"
        )

    # Build message history for LLM
    messages = [SYSTEM_PROMPT]
    for msg in history:
        if msg["role"] in ["user", "assistant"]:
            messages.append(msg)

    # Add current user message for LLM
    messages.append({"role": "user", "content": final_prompt})

    # Get complete response (not streaming chunks)
    complete_response = llm.invoke(messages).content
    print(f"Complete response: {complete_response}")

    # Add assistant response to history
    assistant_message = {"role": "assistant", "content": complete_response}
    final_history = updated_history + [assistant_message]

    # Convert text to speech
    options = KokoroTTSOptions(
        voice="af_heart",
        speed=1.0,
        lang="es"
    )
    
    # Generate audio from complete response
    tts_response = model_tts.tts(complete_response, options=options)
    yield tts_response 
    yield AdditionalOutputs(final_history)  

# endregion

with gr.Blocks() as demo:
    gr.Markdown("# WAIter Chatbot")
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        avatar_images=(
            None,
            "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png",
        ),
    )

    with gr.Row():
        audio = WebRTC(
            mode="send-receive",
            modality="audio",
        )

    # Eventos para audio
    audio.stream(fn=ReplyOnPause(response),
                 inputs=[audio, chatbot],
                 outputs=[audio],
                 time_limit=60)

    audio.on_additional_outputs(lambda a: a,
                                outputs=[chatbot],
                                queue=False)  

if __name__ == "__main__":
    demo.launch()