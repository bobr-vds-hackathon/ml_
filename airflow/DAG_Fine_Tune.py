from airflow import DAG
import datetime
from datetime import datetime, timedelta
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.bash import BashOperator
import os
import cv2
from ultralytics import YOLO
import sys
import shutil

dataset_folder = sys.argv[1]


def process_data():
    data_path = dataset_folder
    labels_path = os.path.join(data_path, "labels/")
    for file in os.listdir(data_path):
        file_label_path = labels_path + str(file.split('.')) + "txt"
        with open(file_label_path, 'w', encoding='utf-8') as f:
            pass
        try:
            img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
            resized = cv2.resize(img, (640, 640), interpolation=cv2.INTER_AREA)
            cv2.imwrite(file, resized)
        except TypeError:
            print("unecpected filetype", flush=True)


def retrain_model():
    model = YOLO('best.pt')
    model.train(data='', patience=10, epochs=60, exist_ok=True)
    pass


def evaluate_model():
    test_data_path = "/path/to/test/data"
    test_data = ""
    model = YOLO()
    predictions = model.predict(test_data)


def clear_data(**kwargs):
    temp_data_path = '/path/to/temp/data'
    shutil.rmtree(temp_data_path)


default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_time': datetime(2023, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=30),
}

with DAG('cv_training_pipeline', default_args=default_args, schedule_interval='@daily', catchup=False) as dag:
    start_task = EmptyOperator(task_id='start')
    preprocess_data_task = PythonOperator(task_id='preprocess_data', python_callable=process_data)
    train_model = PythonOperator(task_id='train', python_callable=retrain_model)
    eval_model = PythonOperator(task_id='eval', python_callable=evaluate_model)
    end_task = EmptyOperator(task_id='end')

    start_task >> preprocess_data_task >> train_model >> eval_model >> end_task
