import tornado.ioloop
import tornado.web


class MainHandler(tornado.web.RequestHandler):
    def get(self):
        self.write("Hello, world")
        self.write(self.request.body)
        self.write("This is me!")

    def post(self):
        for field_name, files in self.request.files.items():
            print(field_name)
            print(files)
            for info in files:
                filename, content_type = info['filename'], info['content_type']
                body = info['body']
                self.write('POST "{}" "{}" "{}" {} bytes\n'.format(field_name, filename, content_type, len(body)))
        self.write('OK\n')


def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])


if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()