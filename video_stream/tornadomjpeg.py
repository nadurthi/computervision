import tornado.ioloop
import tornado.web
import gen
import time
import os
import tornado
import cv2
from PIL import Image
import StringIO

file_list = os.listdir("/home/venkat/Pictures/102D3300")
counter = 0

global capture
capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 320); 
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 240);
capture.set(cv2.CAP_PROP_SATURATION,0.2);
global img

def current_jpeg():
    img = None

    global counter
    counter = (counter + 1)
    print "reading "+str( counter )
    rc,img = capture.read()
    # img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)

    if not rc:
        return None

    tmpFile=cv2.imencode('.jpg', img)[1].tostring()
    return  tmpFile


class MJPEGHandler(tornado.web.RequestHandler):
    @tornado.web.asynchronous
    @tornado.gen.coroutine
    def get(self):
        ioloop = tornado.ioloop.IOLoop.current()

        self.set_header('Cache-Control', 'no-store, no-cache, must-revalidate, pre-check=0, post-check=0, max-age=0')
        self.set_header('Connection', 'close')
        self.set_header( 'Content-Type', 'multipart/x-mixed-replace;boundary=--boundarydonotcross')
        self.set_header('Expires', 'Mon, 3 Jan 2000 12:34:56 GMT')
        self.set_header( 'Pragma', 'no-cache')


        self.served_image_timestamp = time.time()
        my_boundary = "--boundarydonotcross\n"
        while True:
            img = current_jpeg()
            interval = 0.05
            if self.served_image_timestamp + interval < time.time():
                self.write(my_boundary)
                self.write("Content-type: image/jpeg\r\n")
                self.write("Content-length: %s\r\n\r\n" % len(img))
                self.write(str(img))
                self.served_image_timestamp = time.time()
                yield tornado.gen.Task(self.flush)
            else:
                yield tornado.gen.Task(ioloop.add_timeout, ioloop.time() + interval)


application = tornado.web.Application([
    (r"/", MJPEGHandler),
])


if __name__ == "__main__":
    application.listen(8888)
tornado.ioloop.IOLoop.instance().start()