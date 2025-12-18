#!/usr/bin/env python3
"""
Launch script for the Soil Moisture Web Application.
"""

import argparse
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from soilmoisture.web.app import create_app
except ImportError as e:
    logger.error(f"Error importing web application: {e}")
    logger.debug("Please ensure you have installed the required dependencies:")
    logger.debug("pip install -r requirements.txt")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Launch Soil Moisture Web Application')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind to (default: 127.0.0.1)')
    parser.add_argument('--port', type=int, default=5000, help='Port to bind to (default: 5000)')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--data-dir', default='./data', help='Directory for data files')
    parser.add_argument('--output-dir', default='./output', help='Directory for output files')
    parser.add_argument('--upload-dir', default='./uploads', help='Directory for uploaded files')
    
    args = parser.parse_args()
    
    # Set environment variables
    os.environ['DATA_DIR'] = args.data_dir
    os.environ['OUTPUT_DIR'] = args.output_dir
    os.environ['UPLOAD_FOLDER'] = args.upload_dir
    
    if args.debug:
        os.environ['FLASK_DEBUG'] = 'true'
    
    # Create directories if they don't exist
    for directory in [args.data_dir, args.output_dir, args.upload_dir]:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    # Create and run the app
    logger.debug(" Starting Soil Moisture Web Application")
    logger.debug(f" Host: {args.host}")
    logger.debug(f" Port: {args.port}")
    logger.debug(f" Data Directory: {args.data_dir}")
    logger.debug(f" Upload Directory: {args.upload_dir}")
    logger.debug(f" Output Directory: {args.output_dir}")
    logger.debug(f" Debug Mode: {'On' if args.debug else 'Off'}")
    logger.debug()
    logger.debug(f" Access the application at: http://{args.host}:{args.port}")
    logger.debug(" API documentation at: http://{0}:{1}/api-docs".format(args.host, args.port))
    logger.debug("  Health check at: http://{0}:{1}/health".format(args.host, args.port))
    logger.debug()
    logger.debug("Press Ctrl+C to stop the server")
    
    try:
        app = create_app()
        app.run(
            host=args.host,
            port=args.port,
            debug=args.debug,
            threaded=True
        )
    except KeyboardInterrupt:
        logger.debug("\n Shutting down server...")
    except Exception as e:
        logger.error(f" Error starting server: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
