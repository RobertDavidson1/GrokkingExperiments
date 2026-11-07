from .groups.zn import Zn

from .operations.bitwise_and import BitwiseAnd

def select_task(task_type, task_name, n):
    match (task_type,  task_name):
        case ("group", "Zn"):
            return Zn(n)
        case("operation", "bitwise_and"):
            return BitwiseAnd()
