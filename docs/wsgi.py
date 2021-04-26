from flask import Flask


def create_app():
    """
    Creates the docs Flask app
    :return: Flask app
    """
    app = Flask(__name__, static_url_path='/', static_folder='build/html/')

    @app.route('/')
    @app.route('/<path:path>')
    def serve_sphinx_docs(path='build/index.html'):
        return app.send_static_file(path)

    return app


app = create_app()


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True)
