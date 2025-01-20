from clearml import Task
import functools
import inspect
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(input_file):
    return pd.read_csv(input_file)


def clearml_task(project_name, task_name=None, tags=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            automatic_task_name = task_name or func.__name__
            task = Task.init(
                project_name=project_name,
                task_name=automatic_task_name,
                tags=tags or []
            )

            # Логируем параметры с учетом имен
            sig = inspect.signature(func)
            bound_args = sig.bind(*args, **kwargs)
            bound_args.apply_defaults()
            params_dict = bound_args.arguments
            task.connect(params_dict)

            # Логирование типов данных и размеров
            for name, value in params_dict.items():
                task.get_logger().report_text(f'Input Parameter "{name}" Type: {type(value)}')
                if isinstance(value, (np.ndarray, pd.DataFrame)):
                    task.get_logger().report_text(f'Input Parameter "{name}" Shape: {value.shape}')

            try:
                result = func(*args, **kwargs)
                task.get_logger().report_scalar(
                    title='Execution',
                    series='Success',
                    value=1,
                    iteration=0
                )
                # Логирование типа данных и формы результата
                task.get_logger().report_text(f'Result Type: {type(result)}')
                if isinstance(result, (np.ndarray, pd.DataFrame)):
                    task.get_logger().report_text(f'Result Shape: {result.shape}')

                task.upload_artifact(name='processed_data', artifact_object=result)

                if isinstance(result, pd.DataFrame):
                    for col in result.select_dtypes(include=np.number).columns:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        n, bins, patches = ax.hist(result[col], bins=30, label=col, color='#1f77b4', edgecolor='black',
                                                   alpha=0.7)
                        ax.set_title(f'Histogram of {col}', fontsize=14)
                        ax.set_xlabel(col, fontsize=12)
                        ax.set_ylabel('Frequency', fontsize=12)
                        ax.grid(axis='y', alpha=0.7)
                        ax.legend(fontsize=10)
                        task.get_logger().report_matplotlib_figure(
                            title='Histograms',
                            series='Dataset',
                            figure=fig
                        )

                fig, ax = plt.subplots(figsize=(8, 6))
                if isinstance(result, pd.DataFrame) and 'score' in result.columns:
                    ax.plot(result['score'], label='Score', color='#2ca02c')
                    ax.set_title('Score Values Over Index', fontsize=14)
                    ax.set_xlabel('Index', fontsize=12)
                    ax.set_ylabel('Score', fontsize=12)
                    ax.grid(True)
                    ax.legend(fontsize=10)
                    task.get_logger().report_matplotlib_figure(
                        title='Data Plot',
                        series='Dataset',
                        figure=fig
                    )

                # Добавляем скаляр "rows_count"
                if isinstance(result, pd.DataFrame):
                    task.get_logger().report_scalar(
                        title='Data Analysis',
                        series='rows_count',
                        value=len(result),
                        iteration=kwargs.get("iteration", 0),
                    )
                    # Логирование таблицы в Debug Samples
                    task.get_logger().report_table(
                        title='Processed Data Sample',
                        series='Data',
                        table_plot=result.head(5)  # выводим первые 5 строк
                    )

                return result

            except Exception as e:
                task.get_logger().report_scalar(
                    title='Execution',
                    series='Error',
                    value=1,
                    iteration=0
                )
                task.get_logger().report_text(str(e))
                raise

        return wrapper

    return decorator


@clearml_task(
    project_name="ML_Lab_Experiments",
    task_name="Data_Processing",
    tags=["preprocessing", "v1"]
)
def process_data(input_file, threshold=0.5, *args, **kwargs):
    data = load_data(input_file)
    processed_data = data[data['score'] > threshold]
    return processed_data


# Использование декоратора с разными thresholds
thresholds = np.arange(0.1, 1.0, 0.2)
for i, threshold in enumerate(thresholds):
    result = process_data(r'C:\Users\User\Desktop\PythonProject19\dataset.csv', threshold=threshold, iteration=i)
