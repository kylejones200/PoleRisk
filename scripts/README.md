# Utility Scripts

This directory contains utility scripts for setup, deployment, and management tasks.

## Database Management

- **migrate_to_database.py** - Migrate CSV data to SQLite/PostgreSQL database

## Deployment Helpers

- **launch_dashboard.py** - Launch the Streamlit dashboard with proper configuration
- **launch_web_app.py** - Launch the web application with production settings

## Usage

Run scripts from the project root:

```bash
# Launch dashboard
python scripts/launch_dashboard.py

# Launch API server
python scripts/launch_web_app.py

# Migrate data to database
python scripts/migrate_to_database.py --input data.csv --output assessment.db
```

These scripts simplify common deployment and management tasks.

