# 🧑‍🍳 WAIter: Voice Chatbot for Restaurants

WAIter es un chatbot interactivo con capacidades de voz diseñado para escenarios de restaurantes. Simula un camarero amigable y servicial que puede responder a las preguntas habladas de los clientes en tiempo real, utilizando tecnologías de reconocimiento de voz, modelos de lenguaje avanzados y síntesis de voz natural.

## 🛠️ Stack Tecnológico

- **Procesamiento de Voz**:
  - 🎙️ [FastRTC](https://fastrtc.org) para streaming de audio en tiempo real
  - 🔊 [ElevenLabs](https://elevenlabs.io/) para síntesis de voz expresiva (TTS)
  - 🗣️ [Whisper](https://openai.com/research/whisper) (vía Groq) para reconocimiento de voz

- **Inteligencia Artificial de Lenguaje**:
  - 💬 [LangChain](https://www.langchain.com/) para orquestación de LLMs
  - 🔄 [LangGraph](https://github.com/langchain-ai/langgraph) para flujos de trabajo de agentes
  - 🧠 [Gemini 2.5](https://deepmind.google/technologies/gemini/) (vía OpenRouter) para procesamiento de lenguaje natural

- **Gestión de Datos**:
  - 🔍 Embeddings vectoriales con HuggingFace (BAAI/bge-m3)
  - 💾 Datos de menú almacenados en formato Markdown
  - 🗄️ [Supabase](https://supabase.com/) para almacenamiento y procesamiento de pedidos

- **Interfaz**:
  - 🧪 [Gradio](https://www.gradio.app/) para la interfaz web interactiva

## 🚀 Características

- **Interacción por voz en tiempo real** con un asistente virtual de camarero
- **Consulta de información del menú** con respuestas contextuales sobre platos, precios y opciones
- **Procesamiento de pedidos** con capacidad para enviar órdenes directamente a sistemas de cocina
- **Flujo conversacional natural** usando LangGraph para orquestación compleja de agentes
- **Integración de base de conocimientos** con RAG (Retrieval Augmented Generation)
- **Memoria de conversación multi-turno** para respuestas contextuales
- **Gestión de pedidos** con almacenamiento en Supabase y sistema de seguimiento
- **Síntesis de voz expresiva** con ElevenLabs para respuestas naturales y fluidas

## 🚀 Primeros Pasos

1. Clona este repositorio
2. Instala las dependencias: `pip install -r requirements.txt` (requiere Python 3.10+)
3. Crea un archivo `.env` con tus claves API:
   ```
   OPENROUTER_API_KEY=tu_clave_openrouter
   OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
   GROQ_API_KEY=tu_clave_groq
   ELEVENLABS_API_KEY=tu_clave_elevenlabs
   HELICONE_API_KEY=tu_clave_helicone_opcional
   SUPABASE_URL=tu_url_supabase
   SUPABASE_KEY=tu_clave_supabase
   ```
4. Ejecuta la aplicación: `python app.py`
5. Abre la interfaz de Gradio en tu navegador

## 🗂️ Estructura del Proyecto

- `app.py` - Punto de entrada principal con interfaz Gradio
- `agent.py` - Implementación del agente LangGraph para asistente de restaurante
- `model.py` - Gestor para la creación y configuración de modelos de lenguaje
- `tools.py` - Herramientas personalizadas para consulta de menú y envío de pedidos
- `supabase_client.py` - Cliente para operaciones con la base de datos Supabase
- `data/carta.md` - Datos del menú del restaurante en formato Markdown
- `utils/` - Módulos de utilidad:
  - `functions.py` - Funciones auxiliares para manejo de mensajes y modelos
  - `logger.py` - Sistema de registro con colores para depuración
  - `classes.py` - Clases de datos como `Order` para representar pedidos

## 🧠 Cómo Funciona

1. **Reconocimiento de Voz**: FastRTC captura audio del micrófono del usuario y lo envía a la API Whisper de Groq para transcripción
2. **Procesamiento de Lenguaje Natural**: El texto transcrito se pasa al agente LangGraph
3. **Procesamiento RAG**: El agente utiliza búsqueda vectorial para encontrar información relevante del menú
4. **Herramientas del Agente**: El sistema utiliza herramientas especializadas para:
   - Búsqueda de información en el menú (vectores)
   - Envío de pedidos a cocina (integración Supabase)
5. **Generación de Respuesta**: Gemini 2.5 (a través de OpenRouter) genera una respuesta contextual y útil
6. **Síntesis de Voz**: ElevenLabs convierte la respuesta de texto a voz natural
7. **Interfaz Web**: Gradio renderiza la conversación y reproduce la respuesta de audio

## 📦 Requisitos del Sistema

Los requisitos detallados se encuentran en el archivo `requirements.txt`. Las principales dependencias incluyen:

- Python 3.10+
- gradio >= 4.26.0
- langchain >= 0.1.0 (con varios componentes adicionales)
- fastrtc >= 0.6.0
- elevenlabs >= 0.2.24
- groq >= 0.4.0
- supabase >= 2.0.0
- sentence-transformers >= 2.2.2

## 📝 Licencia

[MIT License](LICENSE)

---

🌐 Desarrollado por Roberto - 2024


