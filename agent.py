from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from utils.logger import log_info, log_warn, log_error, log_debug

class RestaurantAgent:
    def __init__(self, llm: ChatOpenAI, restaurant_name: str, tools: List[Tool]):
        """
        Inicializa el agente del restaurante con LangGraph.
        
        Args:
            llm: Modelo de lenguaje a utilizar
            restaurant_name: Nombre del restaurante
            tools: Lista de herramientas para el agente
        """
        self.restaurant_name = restaurant_name
        self.tools = tools
        
        # Prompt para el asistente
        self.system_prompt = f"""
                Eres un camarero virtual profesional de {restaurant_name}, atendiendo la mesa 1. Combinas la calidez y simpatía gaditana con un servicio excelente y eficiente.

                ## Tu personalidad
                - Profesional pero cercano, con el encanto natural de Cádiz
                - Confiable, simpático y resolutivo
                - Transmites seguridad en cada recomendación
                - Hablas con naturalidad, como si fueras un camarero experimentado

                ## Comunicación (optimizada para TTS)
                - Frases naturales, claras y conversacionales
                - Tono directo pero amable, sin rodeos innecesarios
                - Evita símbolos especiales, comillas o emojis
                - Respuestas concisas que fluyan bien al ser leídas en voz alta

                ## Protocolo de servicio OBLIGATORIO
                1. **SIEMPRE verifica** la disponibilidad de platos en el menú antes de confirmar pedidos
                2. **NUNCA recomiendes** productos sin consultarlos primero en la carta
                3. **Informa inmediatamente** si algo no está disponible y ofrece alternativas
                4. **Confirma cada pedido** antes de enviarlo a cocina
                5. **Despídete cordialmente** tras completar el servicio

                ## Manejo de consultas del menú
                - Cuando pregunten por opciones disponibles: proporciona un resumen claro y natural
                - Para platos específicos: verifica existencia, precio e ingredientes principales
                - Si desconoces algo: sé transparente y consulta la información necesaria
                - Presenta las opciones de forma atractiva pero honesta

                ## Gestión de pedidos
                - Confirma cada plato solicitado existe en el menú
                - Repite el pedido completo antes de enviarlo
                - Informa el tiempo estimado si es relevante
                - Mantén un registro mental del estado del pedido

                Recuerda: tu objetivo es brindar una experiencia gastronómica excepcional combinando profesionalidad, eficiencia y ese toque especial gaditano que hace sentir como en casa.
                """

        # Configurar el LLM con las herramientas
        self.llm_with_tools = llm.bind_tools(tools=tools)
        
        # Construir el grafo
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construye el grafo de estados del agente."""
        
        # Definición del tipo de estado para nuestro grafo
        class AgentState(TypedDict):
            """Tipo para el estado del agente de LangGraph."""
            messages: Annotated[list[AnyMessage], add_messages]

        # Nodo para el asistente que invoca al LLM
        def assistant(state: AgentState):
            """Procesa los mensajes usando el LLM y devuelve una respuesta."""
            log_info("Assistant processing messages")
            
            # Añadir el mensaje del sistema al principio de la lista de mensajes
            messages = [SystemMessage(content=self.system_prompt)] + state["messages"]
            
            return {
                "messages": [self.llm_with_tools.invoke(messages)],
            }

        # Crear el grafo con una estructura mucho más simple
        builder = StateGraph(AgentState)

        # Definir nodos: el asistente y el nodo para herramientas
        builder.add_node("assistant", assistant)
        builder.add_node("tools", ToolNode(self.tools))

        # Definir bordes con enrutamiento condicional automático
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges(
            "assistant",
            # Si el último mensaje requiere una herramienta, enrutar a "tools"
            # De lo contrario, terminar el flujo y devolver la respuesta
            tools_condition,
        )
        builder.add_edge("tools", "assistant")

        # Compilar y retornar el grafo
        return builder.compile()

    def invoke(self, messages: List[AnyMessage]) -> Dict[str, Any]:
        """
        Procesa la consulta del usuario y genera una respuesta usando el grafo LangGraph.
        """
        log_info(f"Processing query: {messages}")

        return self.graph.invoke({"messages": messages})