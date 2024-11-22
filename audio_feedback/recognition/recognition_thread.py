from queue import Queue
from threading import Thread


class RecognitionThread(Thread):
    def __init__(self, model, in_queue, *args, **kwargs):
        super(RecognitionThread, self).__init__(*args, **kwargs)

        self.model = model
        self.in_queue = in_queue
        self.out_queue = Queue(maxsize=30)

        self.stop_flag = False

    def run(self):
        while not self.stop_flag:
            color_frame, depth_frame, color_image = self.in_queue.get()
            results = self.model.predict(color_image)
            self.out_queue.put((color_frame, depth_frame, results))

    def stop(self):
        self.stop_flag = True


if __name__ == "__main__":
    recog = RecognitionThread()
