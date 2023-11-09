from airflow import DAG
import datetime
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
import os
import cv2
from ultralytics import YOLO


def download_data(**kwargs):
    pass


def process_data():
    data_path = ""
    labels_path = ""
    for file in os.listdir(data_path):
        file_label_path = labels_path + str(file.split('.')) + "txt"
        with open(file_label_path, 'w', encoding='utf-8') as f:
            pass

        img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
        resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
        cv2.imwrite(file, resized)


def retrain_model(**kwargs):
    model = YOLO('best.pt')
    model.train(data='', patience=10, epochs=60, exist_ok=True)
    pass


def evaluate_model(**kwargs):
    pass


def clear_data(**kwargs):

    pass


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_time': datetime(2023, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
}

with DAG('cv_training_pipeline',default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    start_task = EmptyOperator(task_id='start')
    download_data = PythonOperator(task_id='download_data', python_callable=download_data)
    preprocess_data_task = PythonOperator(task_id='preprocess_data', python_callable=process_data)
    train_model = PythonOperator(task_id='train', python_callable=retrain_model)
    eval_model = PythonOperator(task_id='eval', python_callable=evaluate_model)
    end_task = EmptyOperator(task_id='end')

    start_task >> download_data >> preprocess_data_task >> train_model >> eval_model >> end_task