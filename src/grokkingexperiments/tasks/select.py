from .groups.zn import Zn
from .operations.bitwise_and import BitwiseAnd

def select_task(config):
    task_type = config.task.type
    task_name = config.task.name

    # Match and return instance of correct group / operation
    match (task_type,  task_name):
        case ("group", "Zn"):
            return Zn(config)
        case("operation", "bitwise_and"):
            return BitwiseAnd(config)
