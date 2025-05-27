import gradio as gr
from os import getenv, environ
from dotenv import load_dotenv
import os
from model import ModelManager
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

# Initialize clients and models to None, will be set during runtime
groq_client = None
eleven_client = None
llm = None
waiter_agent = None

# region RAG
md_path = "data/carta.md"

with open(md_path, "r", encoding="utf-8") as file:
    md_content = file.read()

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
        ("#", "seccion_principal"),
        ("##", "categoria"),
    ], 
    strip_headers=False)
splits = splitter.split_text(md_content)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3", model_kwargs = {'device': 'cpu'})
vector_store = InMemoryVectorStore.from_documents(splits, embeddings)

retriever = vector_store.as_retriever(search_kwargs={"k": 4})
# endregion

# Initialize tools to None
guest_info_tool = None
send_to_kitchen_tool = None
tools = None

# Function to initialize all components with provided API keys
def initialize_components(openrouter_key, groq_key, elevenlabs_key, model_name):
    global groq_client, eleven_client, llm, waiter_agent, guest_info_tool, send_to_kitchen_tool, tools
    
    log_info("Initializing components with provided API keys...")
    
    # Initialize clients with provided keys
    if groq_key:
        groq_client = AsyncClient(api_key=groq_key)
    
    if elevenlabs_key:
        eleven_client = ElevenLabs(api_key=elevenlabs_key)
    
    if openrouter_key:
        # Initialize LLM
        model_manager = ModelManager(
            api_key=openrouter_key,
            api_base=getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
            model_name=model_name,
            helicone_api_key=getenv("HELICONE_API_KEY", "")
        )
        llm = model_manager.create_model()
        
        # Initialize tools
        guest_info_tool = create_menu_info_tool(retriever)
        send_to_kitchen_tool = create_send_to_kitchen_tool(llm=llm)
        tools = [guest_info_tool, send_to_kitchen_tool]
        
        # Initialize the agent
        waiter_agent = RestaurantAgent(
            llm=llm,
            tools=tools
        )
        
        log_success("Components initialized successfully.")
    else:
        log_warn("OpenRouter API key is required for LLM initialization.")
    
    return {
        "groq_client": groq_client is not None,
        "eleven_client": eleven_client is not None,
        "llm": llm is not None,
        "agent": waiter_agent is not None
    }

# region FUNCTIONS
async def handle_text_input(message, history, openrouter_key, groq_key, elevenlabs_key, model_name):
    """Handles text input, generates response, updates chat history."""
    global waiter_agent, llm
    
    # Initialize components if needed
    if waiter_agent is None or llm is None or model_name != getattr(llm, "model_name", ""):
        status = initialize_components(openrouter_key, groq_key, elevenlabs_key, model_name)
        if not status["agent"]:
            return history + [
                {"role": "user", "content": message},
                {"role": "assistant", "content": "Error: Could not initialize the agent. Please check your API keys."}
            ]
    
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
        return current_history + [
            {"role": "user", "content": message},
            {"role": "assistant", "content": f"Error: {str(e)}"}
        ]
    
async def response(audio: tuple[int, np.ndarray], history, openrouter_key, groq_key, elevenlabs_key, model_name): 
    """Handles audio input, generates response, yields UI updates and audio."""
    global waiter_agent, llm, groq_client, eleven_client
    
    # Initialize components if needed
    if waiter_agent is None or llm is None or groq_client is None or eleven_client is None or model_name != getattr(llm, "model_name", ""):
        status = initialize_components(openrouter_key, groq_key, elevenlabs_key, model_name)
        if not status["groq_client"]:
            yield AdditionalOutputs(history + [{"role": "assistant", "content": "Error: Groq API key is required for audio processing."}])
            return
        if not status["eleven_client"]:
            yield AdditionalOutputs(history + [{"role": "assistant", "content": "Error: ElevenLabs API key is required for audio processing."}])
            return
        if not status["agent"]:
            yield AdditionalOutputs(history + [{"role": "assistant", "content": "Error: Could not initialize the agent. Please check your OpenRouter API key."}])
            return

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
        await asyncio.sleep(0.04) # Permite que la UI se actualice antes de continuar

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
                    speed=1.1,
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
        yield AdditionalOutputs(current_history + [{"role": "assistant", "content": f"Error: {str(e)}"}])
# endregion

with gr.Blocks() as demo:
    gr.Markdown("# WAIter Chatbot")
    with gr.Row():
        text_openrouter_api_key = gr.Textbox(
            label="OpenRouter API Key (required)",
            placeholder="Enter your OpenRouter API key",
            value=getenv("OPENROUTER_API_KEY") or "",
            type="password",
        )
        text_groq_api_key = gr.Textbox(
            label="Groq API Key (required for audio)",
            placeholder="Enter your Groq API key",
            value=getenv("GROQ_API_KEY") or "",
            type="password",
        )
        text_elevenlabs_api_key = gr.Textbox(
            label="Elevenlabs API Key (required for audio)",
            placeholder="Enter your Elevenlabs API key",
            value=getenv("ELEVENLABS_API_KEY") or "",
            type="password",
        )
        
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
        model_dropdown = gr.Dropdown(
            label="Select Model",
            choices=["google/gemini-2.5-flash-preview-05-20"],
        )

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
        inputs=[
            text_input, 
            chatbot, 
            text_openrouter_api_key, 
            text_groq_api_key, 
            text_elevenlabs_api_key,
            model_dropdown
        ],
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
        inputs=[audio, chatbot, text_openrouter_api_key, text_groq_api_key, text_elevenlabs_api_key, model_dropdown], 
        outputs=[audio], 
    )

    # Actualiza el historial de la conversación
    audio.on_additional_outputs(
        fn=lambda history_update: history_update, # Envia el historial actualizado 
        outputs=[chatbot], # Actualiza el chatbot 
    )

if __name__ == "__main__":
    demo.launch()