from typing import Dict, Any
from datetime import datetime

class Order:
    """Representa un pedido de comida para la cocina."""
    def __init__(self, items=None, special_instructions=None, table_number=None):
        self.items = items or []
        self.special_instructions = special_instructions or ""
        self.table_number = table_number
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.order_id = f"ORD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.error = None  # Campo para indicar si hay un error
    
    def add_item(self, item: Dict[str, Any]):
        """Añade un elemento al pedido."""
        self.items.append(item)
    
    def to_dict(self):
        """Convierte el pedido a un diccionario."""
        return {
            "order_id": self.order_id,
            "timestamp": self.timestamp,
            "table_number": self.table_number,
            "items": self.items,
            "special_instructions": self.special_instructions,
            "error": self.error
        }
    
    def __str__(self):
        """Representación en texto del pedido."""
        if self.error:
            return f"Pedido con error: {self.error}"
            
        items_str = ", ".join([f"{item.get('quantity', 1)}x {item.get('name', 'item')}" for item in self.items])
        return f"Pedido {self.order_id} - Mesa {self.table_number}: {items_str}"