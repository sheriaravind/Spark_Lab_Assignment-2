from tweepy import OAuthHandler
from tweepy import Stream
from tweepy.streaming import StreamListener
import socket
import json
import time

consumer_key = 'LTWzE5RIZrNiBL2NWVpmnq9Rd'
consumer_secret = 'O2OtURmBZHMdpEfyVpkOYVreauB3tJrF8Ww5BPYWqxnCAEJJzg'
access_token = '474841905-z0HE681ecahhBcNngLwf4Bb25vJWGQBw9vi2wON7'
access_secret = '4riBdHwe8DLD4Yqsp56aZLT7QWiNIhiBqPXpQfq0PJHkL'

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)

class TweetsListen(StreamListener):

    def __init__(self, csocket):
        self.client_socket = csocket

    def on_data(self, data):
        try:
            msg = json.loads(data)
            print(msg['text'].encode('utf-8'))
            self.client_socket.send(msg['text'].encode('utf-8'))
            return True
        except BaseException as e:
            print("Error on_data: %s" % str(e))
        return True

    def on_error(self, status):
        print(status)
        return True


def sendData(c_socket):
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)

    twitter_stream = Stream(auth, TweetsListen(c_socket))
    twitter_stream.filter(track=['Football'])


if __name__ == "__main__":
    s = socket.socket()  # Create a socket object
    host = "localhost"  # Get local machine name
    port = 5555  # Reserve a port for your service.
    s.bind((host, port))  # Bind to the port

    print("Listening on port: %s" % str(port))

    s.listen(5)  # Now wait for client connection.
    c, addr = s.accept()  # Establish connection with client.

    print("Received request from: " + str(addr))

    time.sleep(5)

    sendData(c)
