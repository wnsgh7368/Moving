from flask import Flask

def create_app():
    app = Flask(__name__)
    
    from app.routes.detection_routes import detection_bp
    app.register_blueprint(detection_bp)
    
    return app 