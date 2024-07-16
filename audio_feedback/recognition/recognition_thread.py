from queue import Queue
from threading import Thread


class RecognitionThread(Thread):
    def __init__(self, model, *args, **kwargs):
        super(RecognitionThread, self).__init__(*args, **kwargs)

        self.model = model
        self.in_queue = Queue(maxsize=30)
        self.out_queue = Queue(maxsize=30)

        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            frame = self.in_queue.get()
            results = self.model.predict(frame)
            self.out_queue.put(results)

    def stop(self):
        self.stop_flag = True
