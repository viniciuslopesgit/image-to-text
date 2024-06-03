import os
import time
from datetime import datetime, timedelta
import schedule

def delete_old_png_files(directory):
    now = datetime.now()
    half_hour_ago = now - timedelta(minutes=30)
    
    for filename in os.listdir(directory):
        if filename.endswith(".png"):
            file_path = os.path.join(directory, filename)
            file_creation_time = datetime.fromtimestamp(os.path.getctime(file_path))
            
            if file_creation_time < half_hour_ago:
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Error deleting {file_path}: {e}")

def job():
    directory = '/home/apps/pixie-lab/static/images'  # Altere para o caminho da pasta desejada
    delete_old_png_files(directory)

if __name__ == "__main__":
    # Executa o trabalho uma vez no início
    job()

    # Agenda o trabalho para executar a cada 1 hora
    schedule.every(1).hours.do(job)
    
    while True:
        schedule.run_pending()
        time.sleep(1)
