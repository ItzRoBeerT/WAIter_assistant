from langchain.tools import Tool
from langchain_core.vectorstores import VectorStoreRetriever
from typing import Optional

from utils.logger import log_info, log_warn, log_error, log_debug
import json
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from utils.classes import Order

from supabase_client import SupabaseOrderManager
import asyncio

try:
    supabase = SupabaseOrderManager()
except Exception as e:
    log_error(f"Error al inicializar el cliente de Supabase: {e}")
    supabase = None


def create_menu_info_tool(retriever: VectorStoreRetriever) -> Tool:
    """
    Crea una herramienta para extraer información relevante del menú del restaurante.
    
    Args:
        retriever: Un retriever configurado para buscar en la base de conocimiento del menú
        
    Returns:
        Una herramienta de LangChain para consultar información del menú
    """
    def extract_text(query: str) -> str:
        """Extrae texto relevante del menú basado en la consulta."""
        results = retriever.invoke(query)
        log_info("ENTRADA DE EXTRACCIÓN DE TEXTO")
        
        result_texts = []
        if results:
            log_info("\n=== Fragmentos de documento utilizados para la respuesta ===")
            for i, result in enumerate(results):
                log_info(f"Fragmento {i+1}: {result.page_content[:100]}...")
                
                # Comprobar si tiene score (algunos retrievers no incluyen este atributo)
                if hasattr(result, 'score'):
                    log_info(f"Score: {result.score}")
                    
                result_texts.append(result.page_content)
            log_info("=========================================================\n")
            
            # Unir los resultados relevantes
            return "\n\n".join(result_texts)
        else:
            return "Lo siento, no tengo información sobre eso."
    
    return Tool(
    name="guest_info_tool",
    description="""Herramienta para consultar información detallada del menú del restaurante.
        Úsala cuando necesites:
        - Buscar platos específicos y verificar su disponibilidad
        - Consultar precios exactos de productos
        - Obtener información sobre ingredientes, alérgenos o composición de platos
        - Explorar secciones del menú (entrantes, principales, postres, bebidas, etc.)
        - Verificar la existencia de productos antes de recomendarlos
        - Responder preguntas específicas sobre la carta del restaurante
        
        Esta herramienta accede al contenido completo del menú para proporcionar información precisa y actualizada.""",
    func=extract_text,
)

def create_send_to_kitchen_tool(llm: ChatOpenAI) -> Tool:
    """
    Crea una herramienta para enviar pedidos a la cocina.
    
    Args:
        llm: Un modelo de lenguaje para analizar la conversación
        
    Returns:
        Una herramienta de LangChain para enviar pedidos a la cocina
    """
    def extract_order_from_summary(conversation_summary: str) -> Order:
        """
        Usa un LLM para extraer detalles del pedido a partir de un resumen de la conversación.
        
        Args:
            conversation_summary: Resumen de la conversación entre cliente y camarero
            
        Returns:
            Objeto Order con los detalles del pedido extraído
        """
        # Crear mensaje para el LLM
        messages = [
            SystemMessage(content="""
            Eres un asistente especializado en extraer información de pedidos de restaurante.
            Analiza el siguiente resumen de conversación entre un cliente y un camarero.
            Extrae SOLO los elementos del pedido (platos, bebidas, etc.), cantidades, y cualquier instrucción especial.
            Devuelve los resultados en formato JSON entre las etiquetas <order> </order> estrictamente con esta estructura:
            {
                "table_number": número_de_mesa (entero o "desconocida" si no se especifica),
                "items": [
                    {
                        "name": "nombre_del_plato",
                        "quantity": cantidad (entero, por defecto 1),
                        "variations": "variaciones o personalizaciones"
                    },
                    ...
                ],
                "special_instructions": "instrucciones especiales generales"
            }
            No incluyas ninguna otra información o explicación, SOLO el JSON entre las etiquetas.
            """),
            HumanMessage(content=f"Resumen de la conversación: {conversation_summary}")
        ]
        
        # Invocar el LLM para obtener el análisis del pedido
        response = llm.invoke(messages)
        response_text = response.content
        
        # Extraer el JSON de la respuesta usando las etiquetas <order></order>
        try:
            # Buscar contenido entre etiquetas <order> y </order>
            import re
            order_pattern = re.compile(r'<order>(.*?)</order>', re.DOTALL)
            order_match = order_pattern.search(response_text)
            
            if order_match:
                # Extraer el contenido JSON de las etiquetas
                json_str = order_match.group(1).strip()
                order_data = json.loads(json_str)
                
                # Crear objeto Order con los datos extraídos
                return Order(
                    items=order_data.get("items", []),
                    special_instructions=order_data.get("special_instructions", ""),
                    table_number=order_data.get("table_number", "desconocida")
                )
            else:
                # Si no hay etiquetas, reportar error
                log_error("No se encontraron etiquetas <order> en la respuesta del LLM")
                # Devolver un objeto Order vacío con un flag de error
                empty_order = Order(table_number="desconocida")
                empty_order.error = "NO_TAGS_FOUND"
                return empty_order
                    
        except json.JSONDecodeError as e:
            log_error(f"Error al parsear JSON de la respuesta del LLM: {e}")
            log_debug(f"Respuesta problemática: {response_text}")
            empty_order = Order(table_number="desconocida")
            empty_order.error = "JSON_PARSE_ERROR"
            return empty_order
        except Exception as e:
            log_error(f"Error inesperado al procesar la respuesta: {e}")
            log_debug(f"Respuesta completa: {response_text}")
            empty_order = Order(table_number="desconocida")
            empty_order.error = "UNKNOWN_ERROR"
            return empty_order
    
    def send_to_kitchen(conversation_summary: str) -> str:
        """
        Procesa el resumen de la conversación para extraer el pedido y enviarlo a la cocina.
        
        Args:
            conversation_summary: Resumen de la conversación cliente-camarero
            
        Returns:
            Mensaje de confirmación
        """
        try:
            log_info(f"Procesando resumen para enviar pedido a cocina...")
            log_debug(f"Resumen recibido: {conversation_summary}")
            
            # Extraer el pedido a partir del resumen
            order = extract_order_from_summary(conversation_summary)
            
            # Verificar si hay un error en el procesamiento
            if hasattr(order, 'error') and order.error:
                if order.error == "NO_TAGS_FOUND":
                    log_error("No se encontraron las etiquetas <order> en la respuesta del LLM")
                    return "Lo siento, ha ocurrido un problema al procesar su pedido. Por favor, inténtelo de nuevo."
                elif order.error == "JSON_PARSE_ERROR":
                    log_error("Error al analizar el JSON en las etiquetas <order>")
                    return "Ha ocurrido un error técnico al procesar su pedido. ¿Podría repetirlo de otra forma?"
                else:
                    log_error(f"Error desconocido: {order.error}")
                    return "Lo siento, algo salió mal al procesar su pedido. Por favor, inténtelo de nuevo."
            
            # Verificar si hay elementos en el pedido
            if not order.items:
                log_warn("No se identificaron artículos en el pedido")
                return "No se pudo identificar ningún artículo en el pedido. ¿Podría repetir su pedido, por favor?"
            
            # Simular envío a la cocina
            order_dict = order.to_dict()
            log_info(f"ENVIANDO PEDIDO A COCINA: {json.dumps(order_dict, indent=2, ensure_ascii=False)}")
            
            # Aquí iría la integración real con el sistema de la cocina
            # Por ejemplo, enviar a una API, base de datos, etc.
            async def async_send_and_get_result(order):
                return await supabase.send_order(order)
            
            res = asyncio.run(async_send_and_get_result(order))
            if res.get("success"):
                log_info(f"Pedido enviado correctamente a la cocina: {res['order_id']}")
                return f"Su pedido ha sido enviado a la cocina. ID de pedido: {res['order_id']}"
            else:
                log_error(f"Error al enviar el pedido a la cocina: {res.get('error', 'Desconocido')}")
                return "Lo siento, hubo un problema al enviar su pedido a la cocina. ¿Podría intentarlo de nuevo?"

        except Exception as e:
            log_error(f"Error al procesar pedido: {e}")
            log_debug(f"Error detallado: {str(e)}")
            import traceback
            log_debug(traceback.format_exc())
            return "Lo siento, hubo un problema al procesar su pedido. ¿Podría intentarlo de nuevo?"
    
    # Retornar la herramienta configurada con la función send_to_kitchen
    return Tool(
        name="send_to_kitchen_tool",
        description="""
        Envía el pedido completo a la cocina. Usa esta herramienta SOLAMENTE cuando el cliente haya terminado de hacer su pedido 
        completo y esté listo para enviarlo.
        
        Esta herramienta espera recibir un RESUMEN de la conversación que describe los elementos del pedido.
        No envíes la conversación completa, solo un resumen claro de lo que el cliente ha pedido, la mesa, 
        y cualquier instrucción especial relevante.
        """,
        func=send_to_kitchen,
    )
