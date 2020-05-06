import pika
import os

RABBITMQ_ADDR = os.environ.get("RABBITMQ_ADDR") or "127.0.0.1"

def callback(ch, method, properties, body):
    pass

def main(): 
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_ADDR)
    )
    channel = connection.channel()
    channel.queue_declare("gaia_input")

    channel.basic_consume(queue="gaia_input", on_message_callback=callback)
    channel.start_consuming()