import os
from flask import Flask, Blueprint
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def create_app():
    app_mount = os.getenv("LAAB_APP_MOUNT", "/")
    debug_port =  int(os.getenv("LAAB_APP_DEBUG_PORT", "8010"))

    app = Flask(__name__)

    # for access to base templates, macros, styles and script imports
    from .tvastar import tvastar, tvastar_docs
    app.register_blueprint(tvastar.bp, url_prefix=f'{app_mount}/{tvastar.name}')
    app.register_blueprint(tvastar_docs.bp, url_prefix=f'{app_mount}/{tvastar_docs.name}')

    from .apps.laab_report import routes as report_routes
    app.register_blueprint(report_routes.bp, url_prefix=f'{app_mount}/')
    
    return app

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("profiles_root", help="Path to the profiles root directory.")
    parser.add_argument("--port", default="8010", help="Port to run the web application on. Default is 8010.")
    
    args = parser.parse_args()
    if args.profiles_root:
        os.environ["LAAB_PROFILES_ROOT"] = args.profiles_root
    if args.port:
        os.environ["LAAB_APP_DEBUG_PORT"] = args.port
    
    app = create_app()
    app.run(debug=True, port=args.port)
    
if __name__ == '__main__':
    main()