from visionradar.models.database import engine, Base
import visionradar.models.entities  # Load models

def create_tables():
    """
    Creates all SQLite database tables and migrates missing columns.
    """
    Base.metadata.create_all(bind=engine)
    try:
        with engine.connect() as conn:
            from sqlalchemy import text
            # Auto-migrate telemetry_json column if missing
            res = conn.execute(text("PRAGMA table_info(processing_jobs)"))
            cols = [row[1] for row in res.fetchall()]
            if "telemetry_json" not in cols:
                conn.execute(text("ALTER TABLE processing_jobs ADD COLUMN telemetry_json JSON"))
                conn.commit()
    except Exception as e:
        print(f"Migration note: {e}")
    print("VisionRadar Database tables created successfully.")

if __name__ == "__main__":
    create_tables()
