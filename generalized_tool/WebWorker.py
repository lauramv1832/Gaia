import pika
import os
import json
import io
from typing import Optional, List

from generalized_tool.GaiaDMML import *

RABBITMQ_ADDR = os.environ.get("RABBITMQ_ADDR") or "localhost"

@dataclass
class GaiaData:
    hr_png: bytearray
    trimmed_png: bytearray
    distance_png: bytearray
    pm_png: Optional[bytearray]
    correctly_clustered: Optional[int]
    incorrectly_clustered: Optional[int]
    accuracy: Optional[float]
    anomaly: Optional[int]
    cluster_nums: Optional[List[int]]

def run_gaia(csv, db_scan, epsilon, cluster_size):
    csv_file = io.StringIO(csv)
    df = create_df(csv_file)
    distance_bytes = io.BytesIO()
    distance_plot(df, csv_file, distance_bytes)
    hr_bytes = io.BytesIO()
    hr_plots(df, csv_file, hr_bytes)
    trimmed_df = trim_data(df)
    trimmed_bytes = io.BytesIO()
    trimmed_hr(trimmed_df, csv_file, trimmed_bytes)
    pm_bytes = None
    if db_scan:
        # do DBScan stuff
        pm_bytes = io.BytesIO()
        pm_plots(df, trimmed_df, csv_file, cluster_size)
        

def callback(ch, method, properties, body):
    request_info = json.loads(body)

def main(): 
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_ADDR)
    )
    channel = connection.channel()
    channel.queue_declare("gaia_input")

    channel.basic_consume(queue="gaia_input", on_message_callback=callback)
    channel.start_consuming()

if __name__ == '__main__':
    main()