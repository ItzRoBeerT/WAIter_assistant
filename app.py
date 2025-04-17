import gradio as gr
from langchain_openai import ChatOpenAI
from os import getenv
from dotenv import load_dotenv
from groq import AsyncClient 
from fastrtc import WebRTC, ReplyOnPause, audio_to_bytes, get_tts_model, KokoroTTSOptions, AdditionalOutputs
import numpy as np
import asyncio 

load_dotenv()

# Constantes
RESTAURANT = "Bar paco"
SYSTEM_PROMPT = {"role": "system", "content": "Responde como si fueras un camarero amigable, servicial y divertido."}

# Clients
groq_client = AsyncClient() 

# LLM 
llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-2.0-flash-001", 
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer {getenv('HELICONE_API_KEY')}"
        }
    },
)

model_tts = get_tts_model("kokoro")

# region FUNCTIONS
async def response(audio: tuple[int, np.ndarray], history = None): 
    """Handles audio input, generates response, yields UI updates and audio."""

    # Initialize history if None or not a list (robustness)
    current_history = history if isinstance(history, list) else []
    print("-" * 20) # Separator for logs
    print(f"Received audio, current history: {current_history}")

    try:
        # 1. Get transcript from audio
        # Ensure audio_to_bytes handles the input format correctly
        audio_bytes = audio_to_bytes(audio)
        transcript = await groq_client.audio.transcriptions.create(
            file=("audio-file.mp3", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        user_text = transcript.text.strip()

        print(f"Transcription: '{user_text}'")

        # 2. Update history with user message
        user_message = {"role": "user", "content": user_text}
        history_with_user = current_history + [user_message]

        # 3. --- IMMEDIATE UI UPDATE ATTEMPT ---
        print(f"Yielding user message update to UI: {history_with_user}")
        yield AdditionalOutputs(history_with_user)
        # --- Give event loop a chance to process the UI update ---
        #await asyncio.sleep(0.01) # Small delay to yield control

        # 4. Prepare prompt and messages for LLM
        messages_for_llm = [SYSTEM_PROMPT] + history_with_user
     
        final_prompt = f"Eres un camarero para {RESTAURANT}. El cliente dice: {user_text},"+\
        				"Debes generar el texto enfocado a un TTS para que pueda ser reproducido." +\
                        "Evita usar comillas o signos de puntuación, no generes textos muy extensos y no uses emojis." 
        
        messages_for_llm.append({"role": "user", "content": final_prompt}) 

        print(f"Messages for LLM: {messages_for_llm}")

        # 5. Get LLM response
        llm_response = llm.invoke(messages_for_llm)
        
        assistant_text = llm_response.content.strip()

        if not assistant_text:
             print("LLM returned empty response.")
             assistant_text = "Lo siento, no sé cómo responder a eso." 

        print(f"LLM response: '{assistant_text}'")

        # 6. Update history with assistant message
        assistant_message = {"role": "assistant", "content": assistant_text}
        final_history = history_with_user + [assistant_message]

        # 7. Convert text to speech
        print("Generating TTS...")
        options = KokoroTTSOptions(
            voice="af_heart",
            speed=1.0,
            lang="es"
        )
   
        tts_output = model_tts.tts(assistant_text, options=options)
        print(f"TTS output: {tts_output}")
        yield tts_output
        yield AdditionalOutputs(final_history) 
	
    except Exception as e:
        print(f"Error in response function: {e}")
        import traceback
        traceback.print_exc()
        # Yield empty audio and current history in case of error
        yield np.array([]).astype(np.int16).tobytes()
        yield AdditionalOutputs(current_history)

# endregion

with gr.Blocks() as demo:
    gr.Markdown("# WAIter Chatbot")
    chatbot = gr.Chatbot(
        label="Agent",
        type="messages",
        value=[],
        avatar_images=(
            None, # User avatar
            "https://em-content.zobj.net/source/twitter/376/hugging-face_1f917.png", # Assistant
        ),
    )

    with gr.Row():
        audio = WebRTC(
            label="Speak Here",
            mode="send-receive", 
            modality="audio",
        )

    # Event: When audio stream data arrives (managed by ReplyOnPause)
    audio.stream(
        fn=ReplyOnPause(response), 
        inputs=[audio, chatbot], 
        outputs=[audio], 
    )

    # Event: Handle AdditionalOutputs yielded by the response function
    audio.on_additional_outputs(
        fn=lambda history_update: history_update, # Pass the yielded history directly
        outputs=[chatbot], # Update the chatbot component
    )

if __name__ == "__main__":
    demo.launch()
