"""
Módulo de logging para la aplicación wAIter.
Proporciona funciones para imprimir logs en colores según su importancia.
"""

class Colors:
    """Constantes de colores ANSI para los logs en terminal."""
    RESET = "\033[0m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"

def log_info(msg):
    """Log información general (azul cian)."""
    print(f"{Colors.CYAN}[INFO] {msg}{Colors.RESET}")
    
def log_warn(msg):
    """Log advertencias (amarillo)."""
    print(f"{Colors.YELLOW}[WARN] {msg}{Colors.RESET}")
    
def log_error(msg):
    """Log errores (rojo)."""
    print(f"{Colors.RED}[ERROR] {msg}{Colors.RESET}")
    
def log_success(msg):
    """Log éxitos/completados (verde)."""
    print(f"{Colors.GREEN}[SUCCESS] {msg}{Colors.RESET}")
    
def log_debug(msg):
    """Log depuración detallada (magenta)."""
    print(f"{Colors.MAGENTA}[DEBUG] {msg}{Colors.RESET}")