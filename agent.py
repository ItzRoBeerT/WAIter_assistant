from typing import TypedDict, Annotated, List, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import AnyMessage, SystemMessage
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from langchain_openai import ChatOpenAI
from langchain.tools import Tool

from utils.logger import log_info, log_warn, log_error, log_debug
from pathlib import Path

class RestaurantAgent:
    def __init__(self, llm: ChatOpenAI, tools: List[Tool]):
        """
        Inicializa el agente del restaurante con LangGraph.
        
        Args:
            llm: Modelo de lenguaje a utilizar
            tools: Lista de herramientas para el agente
        """
        self.tools = tools
        
        # System Prompt mejorado
        prompt_path = Path("data/system_prompt.txt")
        log_debug(f"Loading system prompt from {prompt_path}")
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

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
        log_info(f"Processing query with Casa Pepe agent: {len(messages)} messages")

        return self.graph.invoke({"messages": messages})