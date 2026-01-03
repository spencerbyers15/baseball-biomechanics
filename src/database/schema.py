"""Database schema initialization and session management."""

import logging
from pathlib import Path
from typing import Generator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from src.database.models import Base

logger = logging.getLogger(__name__)

# Global engine and session factory
_engine: Optional[Engine] = None
_SessionLocal: Optional[sessionmaker] = None


def get_engine(database_url: str = "sqlite:///data/baseball_biomechanics.db", echo: bool = False) -> Engine:
    """
    Get or create the database engine.

    Args:
        database_url: Database connection URL.
        echo: If True, log all SQL statements.

    Returns:
        SQLAlchemy engine instance.
    """
    global _engine

    if _engine is None:
        # Ensure directory exists for SQLite
        if database_url.startswith("sqlite:///"):
            db_path = database_url.replace("sqlite:///", "")
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        _engine = create_engine(
            database_url,
            echo=echo,
            pool_pre_ping=True,  # Verify connections before using
        )

        # Enable foreign keys for SQLite
        if database_url.startswith("sqlite"):
            @event.listens_for(_engine, "connect")
            def set_sqlite_pragma(dbapi_connection, connection_record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()

        logger.info(f"Database engine created: {database_url}")

    return _engine


def get_session_factory(engine: Optional[Engine] = None) -> sessionmaker:
    """
    Get or create the session factory.

    Args:
        engine: Optional engine instance. If not provided, uses global engine.

    Returns:
        SQLAlchemy sessionmaker instance.
    """
    global _SessionLocal

    if _SessionLocal is None:
        if engine is None:
            engine = get_engine()
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.debug("Session factory created")

    return _SessionLocal


def get_session() -> Generator[Session, None, None]:
    """
    Get a database session.

    Yields:
        Database session that will be closed after use.

    Example:
        with get_session() as session:
            players = session.query(Player).all()
    """
    SessionLocal = get_session_factory()
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def init_db(database_url: str = "sqlite:///data/baseball_biomechanics.db", echo: bool = False) -> Engine:
    """
    Initialize the database by creating all tables.

    Args:
        database_url: Database connection URL.
        echo: If True, log all SQL statements.

    Returns:
        SQLAlchemy engine instance.
    """
    engine = get_engine(database_url, echo)
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created successfully")
    return engine


def drop_all_tables(engine: Optional[Engine] = None) -> None:
    """
    Drop all tables from the database.

    WARNING: This is destructive and will delete all data!

    Args:
        engine: Optional engine instance. If not provided, uses global engine.
    """
    if engine is None:
        engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


def reset_engine() -> None:
    """
    Reset the global engine and session factory.

    Useful for testing or when switching databases.
    """
    global _engine, _SessionLocal
    if _engine is not None:
        _engine.dispose()
    _engine = None
    _SessionLocal = None
    logger.debug("Database engine reset")
