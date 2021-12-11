import os
import threading

class WebServer(threading.Thread):
    def __init__(self, *args, **kwargs):
        super(WebServer, self).__init__(*args[1:], **kwargs)
        self.name = args[0]
    
    def run(self):
        os.system('streamlit run server.py --server.address 127.0.0.1')

        
if __name__ == "__main__":
    th_server = WebServer('server')

    threads = list()
    threads.append(th_server)

    exited = list()

    for th in threads:
        th.start()
    
    for th in threads:
        th.join()
        exited.append(th.name)

    string = ''
    for name in exited:
        string += ':%s' % name
    string += ' terminated.'

    print('Program terminated')