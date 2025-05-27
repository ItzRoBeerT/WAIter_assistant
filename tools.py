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
import os
from dotenv import load_dotenv
import re # Asegúrate de importar re

# Cargar variables de entorno
load_dotenv()

# Intentar inicializar Supabase de forma segura
def init_supabase_client():
    """Inicializa el cliente de Supabase de forma segura."""
    try:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if not supabase_url or not supabase_key:
            log_warn("Variables de entorno de Supabase no encontradas. Funcionando sin base de datos.")
            return None
            
        supabase = SupabaseOrderManager()
        log_info("Cliente de Supabase inicializado correctamente")
        return supabase
        
    except Exception as e:
        log_error(f"Error al inicializar el cliente de Supabase: {e}")
        log_warn("Continuando sin funcionalidad de base de datos")
        return None

supabase = init_supabase_client()


def create_menu_info_tool(retriever: VectorStoreRetriever) -> Tool:
    """
    Crea una herramienta para extraer información relevante del menú del restaurante.
    """
    def extract_text(query: str) -> str:
        """Extrae texto relevante del menú basado en la consulta."""
        log_info(f"MENUTOOL: Recibida consulta: '{query}'")
        results = retriever.invoke(query)
        
        result_texts = []
        if results:
            log_info("\n=== MENUTOOL: Fragmentos de documento utilizados para la respuesta ===")
            for i, result in enumerate(results):
                log_info(f"Fragmento {i+1}: {result.page_content[:100]}...")
                if hasattr(result, 'score'):
                    log_info(f"Score: {result.score}")
                result_texts.append(result.page_content)
            log_info("=========================================================\n")
            return "\n\n".join(result_texts)
        else:
            log_info("MENUTOOL: No se encontraron resultados para la consulta.")
            return "Lo siento, no tengo información sobre eso en el menú."
    
    return Tool(
        name="restaurant_menu_lookup_tool",
        description="""
            Herramienta para consultar información detallada sobre el menú del restaurante.
            Esencial para responder preguntas de los clientes sobre platos, ingredientes, precios, alérgenos, opciones dietéticas (vegetarianas, sin gluten, etc.), y disponibilidad de artículos.

            DEBES usar esta herramienta cuando:
            - Un cliente te pide un plato (e.g., "Quiero una tortilla de patatas"). 
            - Un cliente pregunte directamente sobre un plato específico (e.g., "¿Tienen lasaña?", "¿Qué lleva la ensalada César?").
            - Necesites verificar la existencia o detalles de un producto del menú antes de hacer una recomendación o confirmar una elección.
            - Un cliente pregunte por precios (e.g., "¿Cuánto cuesta la hamburguesa?").
            - Un cliente tenga dudas sobre ingredientes o alérgenos (e.g., "¿La paella lleva marisco?", "¿Este postre tiene frutos secos?").
            - Necesites buscar opciones que cumplan ciertos criterios dietéticos o de preferencia (e.g., "platos vegetarianos", "postres sin lactosa", "algo picante").
            - El cliente quiera explorar secciones del menú (e.g., "¿Qué tienen de entrantes?", "¿Qué cervezas ofrecen?").

            Cómo funciona:
            Toma una pregunta o consulta en lenguaje natural sobre el menú como entrada (input) y devuelve la información relevante encontrada en la base de datos del menú como salida (output).
        """,
        func=extract_text,
    )

def create_send_to_kitchen_tool(llm: ChatOpenAI) -> Tool:
    """
    Crea una herramienta para procesar y enviar pedidos a la cocina.
    """
    def extract_order_from_summary(conversation_summary: str) -> Order:
        messages = [
            SystemMessage(content="""
            Eres un asistente experto en extraer información de pedidos de restaurante a partir de un resumen de conversación.
            Analiza el siguiente resumen. Extrae ÚNICAMENTE los artículos del pedido (platos, bebidas), sus cantidades, y cualquier instrucción o variación especial.
            También extrae el número de mesa si está presente.
            Debes devolver los resultados en formato JSON estrictamente entre las etiquetas <order> y </order>.
            El JSON debe seguir esta estructura exacta:
            {
                "table_number": número_de_mesa (entero o la cadena "desconocida" si no se especifica),
                "items": [
                    {
                        "name": "nombre_del_plato_o_bebida",
                        "quantity": cantidad_del_articulo (entero, por defecto 1 si no se especifica),
                        "variations": "variaciones, personalizaciones o notas para este artículo específico" (cadena vacía si no hay)
                    }
                ],
                "special_instructions": "instrucciones especiales generales para todo el pedido" (cadena vacía si no hay)
            }
            Si no puedes identificar ningún artículo o el resumen no parece un pedido, devuelve un JSON con "items" como una lista vacía.
            No incluyas ninguna explicación, saludo o texto adicional fuera de las etiquetas <order> </order>. SOLO el JSON.
            Ejemplo de un buen input de resumen: "Mesa 5, una pizza margarita, dos cocacolas, una sin hielo. La pizza bien hecha."
            Ejemplo de un buen output JSON:
            <order>
            {
                "table_number": 5,
                "items": [
                    {"name": "pizza margarita", "quantity": 1, "variations": "bien hecha"},
                    {"name": "cocacola", "quantity": 2, "variations": "una sin hielo"}
                ],
                "special_instructions": ""
            }
            </order>
            """),
            HumanMessage(content=f"Resumen de la conversación para extraer el pedido: {conversation_summary}")
        ]
        
        response = llm.invoke(messages)
        response_text = response.content
        log_debug(f"KITCHENTOOL_LLM_RESPONSE: {response_text}")
        
        try:
            order_pattern = re.compile(r'<order>(.*?)</order>', re.DOTALL)
            order_match = order_pattern.search(response_text)
            
            if order_match:
                json_str = order_match.group(1).strip()
                log_debug(f"KITCHENTOOL_JSON_EXTRACTED: {json_str}")
                order_data = json.loads(json_str)
                
                # Validaciones adicionales
                if not isinstance(order_data.get("items"), list):
                    log_warn("KITCHENTOOL: 'items' no es una lista o falta en el JSON. Forzando a lista vacía.")
                    order_data["items"] = []

                return Order(
                    items=order_data.get("items", []),
                    special_instructions=order_data.get("special_instructions", ""),
                    table_number=order_data.get("table_number", "desconocida")
                )
            else:
                log_error("KITCHENTOOL: No se encontraron etiquetas <order> en la respuesta del LLM para extraer el pedido.")
                empty_order = Order(table_number="desconocida")
                empty_order.error = "NO_TAGS_FOUND"
                return empty_order
                    
        except json.JSONDecodeError as e:
            log_error(f"KITCHENTOOL: Error al parsear JSON de la respuesta del LLM: {e}")
            empty_order = Order(table_number="desconocida")
            empty_order.error = "JSON_PARSE_ERROR"
            return empty_order
        except Exception as e:
            log_error(f"KITCHENTOOL: Error inesperado al procesar la respuesta del LLM para pedido: {e}")
            empty_order = Order(table_number="desconocida")
            empty_order.error = "UNKNOWN_ERROR_LLM_ORDER_EXTRACTION"
            return empty_order
    
    def send_to_kitchen(conversation_summary_for_order: str) -> str:
        """
        Procesa el resumen de la conversación para extraer el pedido y enviarlo/simularlo.
        """
        try:
            log_info(f"KITCHENTOOL: Iniciando procesamiento de pedido con resumen.")
            log_debug(f"KITCHENTOOL: Resumen recibido para pedido: {conversation_summary_for_order}")
            
            order = extract_order_from_summary(conversation_summary_for_order)
            
            if hasattr(order, 'error') and order.error:
                log_error(f"KITCHENTOOL: Error en la extracción del pedido: {order.error}")
                if order.error == "NO_TAGS_FOUND":
                    return "Lo siento, tuve un problema técnico al intentar entender el pedido. ¿Podrías repetirlo claramente, por favor?"
                elif order.error == "JSON_PARSE_ERROR":
                    return "Lo siento, tuve un problema técnico al procesar los detalles del pedido. ¿Podrías decírmelo de otra manera?"
                return "Lo siento, algo salió mal al procesar el pedido. Por favor, inténtalo de nuevo."
            
            if not order.items:
                log_warn("KITCHENTOOL: No se identificaron artículos en el pedido tras la extracción.")
                return "No pude identificar ningún artículo en tu pedido. ¿Podrías decirme qué te gustaría pedir, por favor?"
            
            order_dict_for_log = order.to_dict() # Para logging
            
            if supabase is None:
                log_warn("KITCHENTOOL: Supabase no configurado. Simulando envío de pedido.")
                log_info(f"PEDIDO PROCESADO (MODO SIMULACIÓN): {json.dumps(order_dict_for_log, indent=2, ensure_ascii=False)}")
                return (f"He procesado tu pedido (en modo simulación ya que la cocina no está conectada ahora mismo). "
                        f"Mesa: {order.table_number}. Artículos: {len(order.items)}. "
                        f"¿Hay algo más en lo que pueda ayudarte?")

            log_info(f"KITCHENTOOL: Enviando pedido a cocina (Supabase): {json.dumps(order_dict_for_log, indent=2, ensure_ascii=False)}")
            
            async def async_send_and_get_result(order_obj):
                return await supabase.send_order(order_obj) # Pasa el objeto Order
            
            res = asyncio.run(async_send_and_get_result(order)) # Pasar el objeto order
            
            if res.get("success"):
                log_info(f"KITCHENTOOL: Pedido enviado correctamente a la cocina. ID: {res['order_id']}")
                return f"¡Perfecto! Tu pedido ha sido enviado a la cocina. El ID de tu pedido es {res['order_id']}. ¿Necesitas algo más?"
            else:
                log_error(f"KITCHENTOOL: Error al enviar el pedido a la cocina vía Supabase: {res.get('error', 'Desconocido')}")
                return "Lo siento, hubo un problema al enviar tu pedido a la cocina. Por favor, intenta confirmarlo de nuevo en un momento."

        except Exception as e:
            log_error(f"KITCHENTOOL: Error general al procesar/enviar pedido: {e}")
            import traceback
            log_debug(traceback.format_exc())
            return "Lo siento, ocurrió un error inesperado al procesar tu pedido. Por favor, inténtalo de nuevo."

    tool_description_base = """
        Procesa y envía el pedido confirmado y finalizado por el cliente a la cocina.
        Utiliza esta herramienta EXCLUSIVAMENTE cuando el cliente haya confirmado verbalmente todos los artículos de su pedido y esté listo para que se tramite. Es el paso final para registrar la orden.

        Qué hace la herramienta:
        1.  Analiza un resumen del pedido proporcionado para extraer: artículos, cantidades, número de mesa e instrucciones especiales.
        2.  Formatea esta información en una orden estructurada.
        3.  {action_description}

        Cuándo DEBES usarla:
        - El cliente dice explícitamente: "Eso es todo", "Listo para pedir", "Envíalo a la cocina", "Confirmo el pedido", o frases similares después de haber detallado todos los artículos de su pedido.
        - Has repasado y confirmado con el cliente la lista completa de artículos y cantidades y el cliente da su aprobación final.

        Qué información necesita como entrada (input):
        - Un RESUMEN CONCISO de la conversación que detalle CLARAMENTE el pedido final. Este resumen DEBE incluir:
            - Lista de artículos (platos, bebidas) con sus respectivas CANTIDADES.
            - Número de MESA (si se especificó o se conoce, de lo contrario se marcará como "desconocida").
            - Cualquier INSTRUCCIÓN ESPECIAL o variación para artículos específicos o para el pedido general (e.g., "sin cebolla en la hamburguesa", "la carne bien hecha", "todo para llevar").
        - NO envíes la transcripción completa de la conversación, solo el resumen del pedido finalizado.
        - NO envíes preguntas sobre el menú a esta herramienta.

        Qué NO hacer:
        - NO la uses si el cliente todavía está explorando el menú, haciendo preguntas sobre platos o añadiendo/modificando artículos. Para consultas sobre el menú, usa 'restaurant_menu_lookup_tool'.
        - NO la uses si el pedido no está completo o el cliente no lo ha confirmado explícitamente.
        - NO la uses para pedir información, solo para enviar un pedido finalizado.
    """

    if supabase is None:
        description = tool_description_base.format(action_description="SIMULA el envío de la orden, ya que la conexión con la cocina no está activa. El pedido se registrará internamente para fines de demostración.")
        description += "\n\nNOTA IMPORTANTE: Actualmente en MODO SIMULACIÓN. El pedido será procesado pero NO se enviará a una cocina real."
    else:
        description = tool_description_base.format(action_description="Envía la orden al sistema real de la cocina.")

    return Tool(
        name="send_order_to_kitchen_tool",
        description=description,
        func=send_to_kitchen,
    )