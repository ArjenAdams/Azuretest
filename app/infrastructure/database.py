from sqlalchemy import create_engine
from sqlalchemy.engine import URL
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

url_object = URL.create(
    "postgresql+psycopg2",
    username="postgres",
    password="postgres",
    host="localhost",
    port=5432,
    database="mimic4",
)

engine = create_engine(url_object, echo=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()
meta = Base.metadata
