from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from typing import List, Dict, Any, Optional
import json
import re



def convert_gradio_to_langchain(current_history: List[Dict[str, Any]], 
                                system_prompt: Optional[Dict[str, Any]] = None) -> List[Any]:
    """
    Convierte el historial de chat de Gradio al formato esperado por LangChain para ReAct agents,
    considerando la estructura específica de la aplicación WAIter Chatbot.
    
    Args:
        current_history: Lista de diccionarios con el formato de Gradio 
                         {"role": "user"|"assistant", "content": str}
        system_prompt: Mensaje del sistema opcional para incluir al inicio
                       {"role": "system", "content": str}
    
    Returns:
        Una lista de objetos de mensaje de LangChain (SystemMessage, HumanMessage, AIMessage, ToolMessage)
    """
    messages_for_agent = []
    
    # Añadir mensaje del sistema si se proporciona
    if system_prompt and system_prompt.get("role") == "system":
        messages_for_agent.append(SystemMessage(content=system_prompt.get("content", "")))
    
    # Mantener registro de las llamadas a herramientas para relacionarlas con sus respuestas
    tool_calls_mapping = {}
    
    for i, msg_dict in enumerate(current_history):
        if msg_dict["role"] == "user":
            messages_for_agent.append(HumanMessage(content=msg_dict["content"]))
        
        elif msg_dict["role"] == "assistant":
            content = msg_dict["content"]
            
            # Buscar patrones de llamadas a herramientas en el texto
            # ReAct pattern: "Action: tool_name\nAction Input: {\"query\": \"...\"}""
            tool_call_pattern = re.search(r'Action: (\w+)\s*\nAction Input:(.+?)(?:\n|$)', content, re.DOTALL)
            
            if "tool_calls" in msg_dict:
                # Caso 1: El mensaje ya incluye tool_calls explícitamente
                ai_message = AIMessage(content=content, tool_calls=msg_dict["tool_calls"])
                messages_for_agent.append(ai_message)
                
                # Guardar referencia para mapear con respuestas de herramientas
                for tool_call in msg_dict["tool_calls"]:
                    tool_calls_mapping[tool_call["id"]] = tool_call
                
            elif tool_call_pattern:
                # Caso 2: Extraer información de herramientas del formato ReAct en el texto
                tool_name = tool_call_pattern.group(1).strip()
                tool_input_str = tool_call_pattern.group(2).strip()
                
                try:
                    # Intentar parsear el input JSON
                    tool_input = json.loads(tool_input_str)
                except json.JSONDecodeError:
                    # Si no es JSON válido, usar como string
                    tool_input = {"input": tool_input_str}
                
                # Crear una estructura de tool_calls compatible
                tool_call_id = f"{tool_name}_{i}"
                tool_calls = [{
                    "id": tool_call_id,
                    "name": tool_name,
                    "args": tool_input
                }]
                
                # Guardar en el mapa para relacionar con posibles respuestas
                tool_calls_mapping[tool_call_id] = tool_calls[0]
                
                # Crear el mensaje con la llamada a herramienta
                ai_message = AIMessage(content=content, tool_calls=tool_calls)
                messages_for_agent.append(ai_message)
                
            else:
                # Caso 3: Mensaje normal sin llamadas a herramientas
                messages_for_agent.append(AIMessage(content=content))
        
        elif msg_dict["role"] == "tool":
            # Manejar respuestas de herramientas
            tool_name = msg_dict.get("name", "unknown_tool")
            tool_content = msg_dict.get("content", "")
            
            # Buscar el ID de la llamada correspondiente
            tool_call_id = None
            for id, call in tool_calls_mapping.items():
                if call["name"] == tool_name:
                    tool_call_id = id
                    break
            
            if not tool_call_id:
                # Si no encontramos coincidencia exacta, usar el nombre como ID
                tool_call_id = tool_name
            
            # Añadir el mensaje de la herramienta
            messages_for_agent.append(
                ToolMessage(content=tool_content, tool_call_id=tool_call_id)
            )
    
    return messages_for_agent


def extract_final_text(agent_output: Any) -> str:
    """
    Extrae el texto final de la respuesta del LLM para ser presentada al usuario.
    Optimizado para el formato de respuesta de LangChain con herramientas.
    """
    # Caso directo: si es un string, devolver directamente
    if isinstance(agent_output, str):
        return agent_output.strip()
    
    # Caso para respuestas de agentes con mensajes
    if isinstance(agent_output, dict) and "messages" in agent_output:
        messages = agent_output["messages"]
        
        # Buscar el último mensaje AI sin llamadas a herramientas
        for msg in reversed(messages):
            if hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage":
                # Verificar si tiene contenido y no tiene tool_calls
                if hasattr(msg, "content") and msg.content and not getattr(msg, "tool_calls", None):
                    return msg.content.strip()
        
        # Fallback: tomar el último mensaje AI incluso si tiene tool_calls
        for msg in reversed(messages):
            if hasattr(msg, "__class__") and msg.__class__.__name__ == "AIMessage":
                if hasattr(msg, "content") and msg.content:
                    return msg.content.strip()
    
    # Para cualquier otro formato o fallo, devolver mensaje genérico
    return "Lo siento, no pude procesar la respuesta correctamente."
