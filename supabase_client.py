from supabase import create_client
import os
from dotenv import load_dotenv
from typing import Dict, List, Any, Optional
import json
import logging

from utils.classes import Order
# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SupabaseOrderManager:
    def __init__(self, url: Optional[str] = None, key: Optional[str] = None):
        """
        Inicializa el gestor de órdenes con Supabase.
        
        Args:
            url: URL de Supabase (opcional, por defecto usa variables de entorno)
            key: Clave de API de Supabase (opcional, por defecto usa variables de entorno)
        """
        # Cargar variables de entorno
        load_dotenv()
        
        # Usar parámetros proporcionados o variables de entorno
        self.supabase_url = url or os.getenv("SUPABASE_URL")
        self.supabase_key = key or os.getenv("SUPABASE_KEY")
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError("SUPABASE_URL y SUPABASE_KEY deben estar definidos como variables de entorno o proporcionados como parámetros")
        
        # Inicializar cliente
        self.supabase = create_client(self.supabase_url, self.supabase_key)
        logger.info("Cliente Supabase inicializado correctamente")
    
    async def send_order(self, order: Order) -> Dict[str, Any]:
        """
        Envía una orden a Supabase.
        
        Args:
            order: Objeto Order del sistema del restaurante
            
        Returns:
            Diccionario con el resultado de la operación
        """
        try:
            # Convertir el objeto Order a un diccionario
            order_dict = order.to_dict()
            
            # 1. Insertar la orden principal
            order_data = {
                "order_id": order_dict["order_id"],
                "table_number": str(order_dict["table_number"]),
                "special_instructions": order_dict["special_instructions"],
                "status": "pending"
            }
            
            # Realizar la inserción de la orden
            order_response = self.supabase.table("orders").insert(order_data).execute()
            
            if not order_response.data:
                raise Exception("Error al insertar orden en Supabase")
            
            # Obtener el ID de la orden insertada
            db_order_id = order_response.data[0]["id"]
            
            # 2. Insertar los elementos de la orden
            items_to_insert = []
            for item in order_dict["items"]:
                items_to_insert.append({
                    "order_id": db_order_id,
                    "name": item["name"],
                    "quantity": item.get("quantity", 1),
                    "variations": item.get("variations", "")
                })
            
            # Realizar la inserción de los elementos
            if items_to_insert:
                items_response = self.supabase.table("order_items").insert(items_to_insert).execute()
                
                if not items_response.data:
                    # Si falla la inserción de items, eliminamos la orden para mantener consistencia
                    self.supabase.table("orders").delete().eq("id", db_order_id).execute()
                    raise Exception("Error al insertar elementos de la orden en Supabase")
            
            logger.info(f"Orden {order_dict['order_id']} enviada correctamente a Supabase")
            return {
                "success": True,
                "order_id": order_dict["order_id"],
                "database_id": db_order_id
            }
            
        except Exception as e:
            logger.error(f"Error al enviar orden a Supabase: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    