import gradio as gr
from langchain_openai import ChatOpenAI
from os import getenv, environ
from dotenv import load_dotenv
import os

# Configurar la variable de entorno para evitar advertencias de tokenizers (huggingface opcional)
os.environ["TOKENIZERS_PARALLELISM"] = "false"
 
from groq import AsyncClient 
from fastrtc import WebRTC, ReplyOnPause, audio_to_bytes, AdditionalOutputs
import numpy as np
import asyncio 
from elevenlabs.client import ElevenLabs
from elevenlabs import VoiceSettings

from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.messages import HumanMessage, AIMessage

from agent import RestaurantAgent
# Importar las herramientas
from tools import create_menu_info_tool, create_send_to_kitchen_tool

from utils.logger import log_info, log_warn, log_error, log_success, log_debug

load_dotenv()

# Constantes
RESTAURANT = "Bar paco"

# Clients
groq_client = AsyncClient() 

eleven_client = ElevenLabs(
  api_key=getenv("ELEVENLABS_API_KEY"),
)

# LLM 
llm = ChatOpenAI(
    openai_api_key=getenv("OPENROUTER_API_KEY"),
    openai_api_base=getenv("OPENROUTER_BASE_URL"),
    model_name="google/gemini-2.5-flash-preview", 
    model_kwargs={
        "extra_headers": {
            "Helicone-Auth": f"Bearer {getenv('HELICONE_API_KEY')}"
        }
    },
)

# region RAG
md_path = "data/carta.md"

with open(md_path, "r", encoding="utf-8") as file:
    md_content = file.read()

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "seccion_principal"),
        ("##", "subseccion"),
        ("###", "apartado")
    ], 
    strip_headers=False)
splits = splitter.split_text(md_content)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_store = InMemoryVectorStore.from_documents(splits, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 3})
# endregion

# region TOOLS
guest_info_tool = create_menu_info_tool(retriever)
send_to_kitchen_tool = create_send_to_kitchen_tool(llm=llm)
tools = [guest_info_tool, send_to_kitchen_tool]
# endregion

# region LANGGRAPH IMPLEMENTATION
# Crear la instancia del agente de restaurante
waiter_agent = RestaurantAgent(
    llm=llm,
    restaurant_name=RESTAURANT,
    tools=tools
)
# endregion

# region FUNCTIONS
async def handle_text_input(message, history):
    """Handles text input, generates response, updates chat history."""
    
    current_history = history if isinstance(history, list) else []
    log_info("-" * 20) 
    log_info(f"Received text input: '{message}', current history: {current_history}")

    try:
        # 1. Actualizar el historial con el mensaje del usuario
        user_message = {"role": "user", "content": message}
        history_with_user = current_history + [user_message]

        # 2. Invocar al agente con la consulta del usuario
        log_info("Iniciando procesamiento con LangGraph...")
        
        # Invocar el agente con el texto de la consulta
        langchain_messages = []
        for msg in current_history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))
    
        langchain_messages.append(HumanMessage(content=message))

        graph_result = waiter_agent.invoke(langchain_messages)
        
        log_debug(f"Resultado del agente: {graph_result}")
        
       
        messages = graph_result.get("messages", [])
        assistant_text = ""
        
        for msg in reversed(messages):
            # LangChain puede devolver diferentes clases de mensajes
            if hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage":
                assistant_text = msg.content
                break
        
        if not assistant_text:
            log_warn("No se encontró respuesta del asistente en los mensajes.")
            assistant_text = "Lo siento, no sé cómo responder a eso."

        log_info(f"Assistant text: '{assistant_text}'")

        # 3. Actualizar el historial con el mensaje del asistente
        assistant_message = {"role": "assistant", "content": assistant_text}
        final_history = history_with_user + [assistant_message]
        
        log_success("Tarea completada con éxito.")
        return final_history

    except Exception as e:
        log_error(f"Error in handle_text_input function: {e}")
        import traceback
        traceback.print_exc()
        return current_history
    
async def response(audio: tuple[int, np.ndarray], history = None): 
    """Handles audio input, generates response, yields UI updates and audio."""

    current_history = history if isinstance(history, list) else []
    log_info("-" * 20)
    log_info(f"Received audio, current history: {current_history}")

    try:
        # 1. Transcribir el audio a texto
        audio_bytes = audio_to_bytes(audio)
        transcript = await groq_client.audio.transcriptions.create(
            file=("audio-file.mp3", audio_bytes),
            model="whisper-large-v3-turbo",
            response_format="verbose_json",
        )
        user_text = transcript.text.strip()

        log_info(f"Transcription: '{user_text}'")

        # 2. Actualizar el historial con el mensaje del usuario
        user_message = {"role": "user", "content": user_text}
        history_with_user = current_history + [user_message]

        log_info(f"Yielding user message update to UI: {history_with_user}")
        yield AdditionalOutputs(history_with_user)
        await asyncio.sleep(0.01) # Permite que la UI se actualice antes de continuar

        # 4. Invocar al agente con la consulta del usuario
        log_info("Iniciando procesamiento con LangGraph...")
        
        langchain_messages = []
        for msg in current_history:
            if msg["role"] == "user":
                langchain_messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                langchain_messages.append(AIMessage(content=msg["content"]))

        langchain_messages.append(HumanMessage(content=user_text))

        graph_result = waiter_agent.invoke(langchain_messages)
        
        log_debug(f"Resultado del agente: {graph_result}")
        
        # Extraer la respuesta del último mensaje del asistente
        messages = graph_result.get("messages", [])
        assistant_text = ""
        
        # Buscar el último mensaje del asistente
        for msg in reversed(messages):
            if hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage":
                assistant_text = msg.content
                break
        
        if not assistant_text:
            log_warn("No se encontró respuesta del asistente en los mensajes.")
            assistant_text = "Lo siento, no sé cómo responder a eso."

        log_info(f"Assistant text: '{assistant_text}'")

        # 5. Actualizar el historial con el mensaje del asistente
        assistant_message = {"role": "assistant", "content": assistant_text}
        final_history = history_with_user + [assistant_message]

        # 6. Generar la respuesta de voz
        log_info("Generating TTS...")
        TARGET_SAMPLE_RATE = 24000 # <<< --- Tasa de muestreo deseada
        tts_stream_generator = eleven_client.text_to_speech.convert(
                text=assistant_text,
                voice_id="Nh2zY9kknu6z4pZy6FhD",
                model_id="eleven_flash_v2_5",
                output_format="pcm_24000",
                voice_settings=VoiceSettings(
                    stability=0.0,
                    similarity_boost=1.0, 
                    style=0.0,
                    use_speaker_boost=True,
                    speed=1.0,
                )
            )
        
        # --- Procesar los chunks a medida que llegan ---
        log_info("Receiving and processing TTS audio chunks...")
        audio_chunks = []
        total_bytes = 0

        for chunk in tts_stream_generator:
            total_bytes += len(chunk)
            
            # Convertir chunk actual de bytes PCM (int16) a float32 normalizado
            if chunk:
                audio_int16 = np.frombuffer(chunk, dtype=np.int16)
                audio_float32 = audio_int16.astype(np.float32) / 32768.0
                audio_float32 = np.clip(audio_float32, -1.0, 1.0)  # Asegurar rango
                audio_chunks.append(audio_float32)

        log_info(f"Received {total_bytes} bytes of TTS audio in total.")

        # Concatenar todos los chunks procesados
        if audio_chunks:
            final_audio = np.concatenate(audio_chunks)
            log_info(f"Processed {len(final_audio)} audio samples.")
        else:
            log_warn("Warning: TTS returned empty audio stream.")
            final_audio = np.array([], dtype=np.float32)
 
        # Crear la tupla final
        tts_output_tuple = (TARGET_SAMPLE_RATE, final_audio)
    
        log_debug(f"TTS output: {tts_output_tuple}")   
        log_success("Tarea completada con éxito.")
        yield tts_output_tuple
        yield AdditionalOutputs(final_history) 
	
    except Exception as e:
        log_error(f"Error in response function: {e}")
        import traceback
        traceback.print_exc()
      
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
        text_input = gr.Textbox(
            label="Type your message",
            placeholder="Type here and press Enter...",
            show_label=True,
        )
        audio = WebRTC(
            label="Speak Here",
            mode="send-receive", 
            modality="audio",
        )
    
    text_input.submit(
        fn=handle_text_input,
        inputs=[text_input, chatbot],
        outputs=[chatbot],
        api_name="submit_text"
    ).then(
        fn=lambda: "",  # Limpiar el campo de texto
        outputs=[text_input]
    )

    # Se encarga de manejar la entrada de audio
    audio.stream(
        fn=ReplyOnPause(
            response, 
            can_interrupt=True,
        ), 
        inputs=[audio, chatbot], 
        outputs=[audio], 
    )

    # Actualiza el historial de la conversación
    audio.on_additional_outputs(
        fn=lambda history_update: history_update, # Envia el historial actualizado 
        outputs=[chatbot], # Actualiza el chatbot 
    )

if __name__ == "__main__":
    demo.launch()