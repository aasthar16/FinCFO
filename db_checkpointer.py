"""
PostgreSQL Checkpointer for LangGraph
Production-ready checkpoint management with connection pooling.
"""

import os
from typing import Optional, Dict, Any
from contextlib import contextmanager
import logging

import psycopg
from psycopg import Connection, sql
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool
from langgraph.checkpoint.postgres import PostgresSaver

logger = logging.getLogger(__name__)

# Global connection pool
_pool: Optional[ConnectionPool] = None


def get_connection_string() -> str:
    """Build PostgreSQL connection string from environment variables."""
    host = os.getenv("PGHOST", "localhost")
    dbname = os.getenv("PGDATABASE", "ai_cfo")
    user = os.getenv("PGUSER", "postgres")
    password = os.getenv("PGPASSWORD", "")
    sslmode = os.getenv("PGSSLMODE", "require")
    
    # Build connection string
    conn_str = f"host={host} dbname={dbname} user={user} password={password}"
    if sslmode:
        conn_str += f" sslmode={sslmode}"
    
    return conn_str


def init_connection_pool(min_size: int = 1, max_size: int = 10) -> ConnectionPool:
    """Initialize the PostgreSQL connection pool."""
    global _pool
    
    if _pool is not None:
        return _pool
    
    conn_str = get_connection_string()
    
    _pool = ConnectionPool(
        conn_str,
        min_size=min_size,
        max_size=max_size,
        timeout=30,
        open=True,
        kwargs={
            "autocommit": True,  # Required for CREATE INDEX CONCURRENTLY
            "row_factory": dict_row,
            "prepare_threshold": None,
        },
    )
    
    logger.info(f"PostgreSQL connection pool initialized: {min_size}-{max_size} connections")
    return _pool


@contextmanager
def get_connection() -> Connection:
    """Get a connection from the pool as a context manager."""
    pool = init_connection_pool()
    with pool.connection() as conn:
        yield conn


def init_checkpointer_tables() -> None:
    """
    Initialize LangGraph checkpoint tables in PostgreSQL.
    This must be run once before using the checkpointer.
    """
    pool = init_connection_pool()
    
    # Get a connection (autocommit is True)
    with pool.connection() as conn:
        # Create the checkpointer tables using LangGraph's internal setup
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        logger.info("LangGraph checkpoint tables initialized")
        
        # Verify tables exist
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'checkpoints'
                );
            """)
            exists = cur.fetchone()['exists']
            
            if exists:
                logger.info("Checkpoint tables verified")
            else:
                logger.warning("Checkpoint tables not found after setup")


def verify_tables_exist() -> bool:
    """Verify that checkpoint tables exist."""
    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables 
                        WHERE table_schema = 'public' 
                        AND table_name = 'checkpoints'
                    );
                """)
                result = cur.fetchone()
                return result['exists'] if result else False
    except Exception as e:
        logger.error(f"Failed to verify tables: {e}")
        return False


def get_checkpointer() -> Optional[PostgresSaver]:
    """
    Get the PostgresSaver checkpointer instance.
    Returns None if database is not available.
    """
    try:
        pool = init_connection_pool()
        with pool.connection() as conn:
            checkpointer = PostgresSaver(conn)
            return checkpointer
    except Exception as e:
        logger.error(f"Failed to create checkpointer: {e}")
        return None


def cleanup_pool() -> None:
    """Close the connection pool."""
    global _pool
    if _pool:
        _pool.close()
        _pool = None
        logger.info("PostgreSQL connection pool closed")


# Initialize on import
init_connection_pool()


if __name__ == "__main__":
    # Run setup when module is executed directly
    import sys
    
    try:
        init_checkpointer_tables()
        print("✅ Checkpointer tables initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize tables: {e}")
        sys.exit(1)