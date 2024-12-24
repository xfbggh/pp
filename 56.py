import yaml
from clearml import Task

def load_params_from_yaml(yaml_file):
    with open(yaml_file, 'r') as file:
        return yaml.safe_load(file)

def cml_task(yaml_file=None, project_name=None, task_name=None, tags=None, artifacts=None):
    def decorator(func):
        def wrapper(*args, **kwargs):
            nonlocal project_name, task_name, tags, artifacts  # Указываем, что эти переменные не локальные

            # Загружаем параметры из YAML, если представлен файл
            if yaml_file:
                config_params = load_params_from_yaml(yaml_file)
                print(config_params)
                project_name = config_params.get('project_name', project_name)
                task_name = config_params.get('task_name', task_name or func.__name__)
                tags = config_params.get('tags', tags)
                artifacts = config_params.get('artifacts', artifacts)  # Артефакты из YAML

            # Инициализация задачи
            task = Task.init(
                project_name=project_name,
                task_name=task_name,
                tags=tags or []
            )

            # Загружаем артефакты, если они были указаны
            if artifacts:
                for artifact_name, artifact_value in artifacts.items():
                    task.upload_artifact(name=artifact_name, artifact_object=artifact_value)

            try:
                # Выполнение основной функции
                result = func(*args, **kwargs)
            except Exception as e:
                # Логируем ошибку
                task.get_logger().report_text(f"Error: {str(e)}")
                raise
            else:
                # Логируем успешное выполнение
                task.get_logger().report_text("задача выполнена")
                return result
            finally:
                # Завершаем работу задачи
                task.close()

        return wrapper
    return decorator



@cml_task(yaml_file='./testConfig.yaml')
def example_function():
    print("задача запущена")


# Вызов функции
example_function()