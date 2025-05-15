from flask import Flask
from flask_cors import CORS
from src.routes.predict import predict_bp

app = Flask(__name__)
CORS(app)

# Register blueprint
app.register_blueprint(predict_bp, url_prefix="/api")

if __name__ == "__main__":
    app.run(debug=True)