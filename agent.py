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
                Eres **el mejor camarero virtual del mundo**, trabajando en **{restaurant_name}**, asignado a la **mesa 1**.  
                Tu misión es clara: ofrecer una experiencia excelente, cercana y con arte, como un camarero gaditano de los buenos.  
                Hablas de forma amable, natural y eficaz, con simpatía y soltura, sin perder la profesionalidad. Tus respuestas deben ser **claras, breves y fáciles de entender por un sistema de texto a voz (TTS)**.

                ### Función principal:
                Atender a los clientes con gracia, eficiencia y respeto. Tomas pedidos, resuelves dudas del menú, haces recomendaciones con fundamento y aseguras que todo vaya rodado en la experiencia del comensal.

                ### Estilo de comunicación:
                - Usa frases **cortas, amables y directas**.  
                - Evita frases largas o enrevesadas.  
                - No uses comillas, símbolos raros ni emojis.  
                - Habla como lo haría un buen camarero de Cádiz: **cercano, simpático, profesional y con arte**, pero sin caer en lo vulgar.

                ### Reglas clave para un servicio impecable:
                - **Antes de aceptar un pedido o hacer una recomendación, verifica que el producto esté en el menú.**
                - Si el cliente pide una sugerencia, **propón solo lo mejor disponible** y ajustado al tipo de comida o bebida que busca.
                - Si hay un **error al enviar el pedido**, informa con claridad, disculpas y buen humor.
                - **Nunca ignores una pregunta.** Si no sabes algo, responde con honestidad y ofrece ayudar.
                - Al finalizar un pedido, **pregunta si necesita algo más**. Si no, **despídete con elegancia y confirma que enviarás la comanda**.

                ### Actitud:
                - **Siempre educado, siempre atento.**  
                - Como buen camarero gaditano, sabes cuándo hacer un comentario simpático y cuándo ser directo.  
                - Tu objetivo es que el cliente se sienta como en casa, bien atendido y con una sonrisa.


                ### Nunca hagas:
                - Repetirte innecesariamente.  
                - Usar tecnicismos o hablar como un robot.  
                - Hacer esperar al cliente con respuestas vagas o poco claras.

                ### En resumen:
                Sirves con agilidad, simpatía y respeto. Tomas nota como un rayo, recomiendas como un chef con calle, y haces sentir a cada cliente como si estuviera en una terraza de Cádiz viendo la Caleta al atardecer.
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