"""
PostgreSQL Checkpointer for LangGraph using psycopg3.
"""

import logging
import psycopg
from psycopg.rows import dict_row

from settings import settings
import json

# Set logger to only show warnings and errors
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


def get_connection():
    """Get a PostgreSQL connection using psycopg3."""
    try:
        if not settings.database_url:
            raise ValueError("DATABASE_URL not configured")
        
        conn = psycopg.connect(
            settings.database_url,
            autocommit=True,
            row_factory=dict_row,
        )
        return conn
    except Exception as e:
        logger.error(f"Failed to connect to database: {e}")
        raise

from langgraph.checkpoint.memory import MemorySaver

_memory_checkpointer = MemorySaver()

def get_checkpointer():
    return _memory_checkpointer

def init_tables():
    """Initialize all database tables."""
    try:
        conn = get_connection()
        
        # Initialize LangGraph checkpoint tables
        checkpointer = PostgresSaver(conn)
        checkpointer.setup()
        
        # Create users table
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    email TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    password_hash TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
        
        # Create user_chats table
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_chats (
                    id SERIAL PRIMARY KEY,
                    email TEXT NOT NULL REFERENCES users(email) ON DELETE CASCADE,
                    chat_name TEXT NOT NULL,
                    messages JSONB NOT NULL DEFAULT '[]'::jsonb,
                    created_at TIMESTAMP DEFAULT NOW(),
                    updated_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(email, chat_name)
                );
            """)
        
        # Create user_profiles table
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS user_profiles (
                    email TEXT PRIMARY KEY REFERENCES users(email) ON DELETE CASCADE,
                    startup_name TEXT,
                    stage TEXT,
                    currency TEXT DEFAULT 'USD',
                    industry TEXT,
                    country TEXT,
                    founded_date DATE,
                    updated_at TIMESTAMP DEFAULT NOW()
                );
            """)
        
        # Create indexes
        with conn.cursor() as cur:
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_chats_email ON user_chats(email);
                CREATE INDEX IF NOT EXISTS idx_user_chats_updated ON user_chats(updated_at DESC);
            """)
        
        conn.close()
        print("✅ Database tables initialized successfully")
        
    except Exception as e:
        print(f"❌ Failed to initialize tables: {e}")
        raise


def verify_tables():
    """Verify that all tables exist."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'checkpoints'
                );
            """)
            checkpoints_exist = cur.fetchone()['exists']
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'users'
                );
            """)
            users_exist = cur.fetchone()['exists']
            
            cur.execute("""
                SELECT EXISTS (
                    SELECT FROM information_schema.tables 
                    WHERE table_schema = 'public' 
                    AND table_name = 'user_profiles'
                );
            """)
            profiles_exist = cur.fetchone()['exists']
            
        conn.close()
        return checkpoints_exist and users_exist and profiles_exist
    except Exception as e:
        logger.error(f"Failed to verify tables: {e}")
        return False


# User Management Functions
def create_user_in_db(email: str, name: str, password_hash: str) -> bool:
    """Create a new user."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO users (email, name, password_hash)
                VALUES (%s, %s, %s)
            """, (email, name, password_hash))
        conn.close()
        print(f"✅ User created: {email}")
        return True
    except Exception as e:
        print(f"❌ Failed to create user: {e}")
        return False


def get_user_from_db(email: str):
    """Get user from database."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT email, name, password_hash, created_at
                FROM users
                WHERE email = %s
            """, (email,))
            user = cur.fetchone()
        conn.close()
        return user
    except Exception as e:
        print(f"❌ Failed to get user: {e}")
        return None


def save_chat_to_db(email: str, chat_name: str, messages: list):
    """Save chat to database."""
    try:
        # Ensure messages is serializable
        messages_json = json.dumps(messages)
        
        conn = get_connection()
        with conn.cursor() as cur:
            # Check if chat exists
            cur.execute("""
                SELECT id FROM user_chats 
                WHERE email = %s AND chat_name = %s
            """, (email, chat_name))
            existing = cur.fetchone()
            
            if existing:
                cur.execute("""
                    UPDATE user_chats 
                    SET messages = %s::jsonb, updated_at = NOW()
                    WHERE email = %s AND chat_name = %s
                """, (messages_json, email, chat_name))
            else:
                cur.execute("""
                    INSERT INTO user_chats (email, chat_name, messages)
                    VALUES (%s, %s, %s::jsonb)
                """, (email, chat_name, messages_json))
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to save chat: {e}")
        return False


def get_user_chats_from_db(email: str) -> list:
    """Get all chats for a user."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT chat_name, messages, created_at, updated_at
                FROM user_chats
                WHERE email = %s
                ORDER BY updated_at DESC
            """, (email,))
            rows = cur.fetchall()
        conn.close()
        
        chats = []
        for row in rows:
            messages = row['messages']
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except:
                    messages = []
            elif not isinstance(messages, list):
                messages = []
            
            # Validate each message
            valid_messages = []
            for msg in messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    valid_messages.append(msg)
            
            chats.append({
                "name": row['chat_name'],
                "messages": valid_messages,
                "created_at": row['created_at'].isoformat() if row['created_at'] else None,
                "updated_at": row['updated_at'].isoformat() if row['updated_at'] else None,
            })
        return chats
    except Exception as e:
        print(f"❌ Failed to get chats: {e}")
        return []


def delete_chat_from_db(email: str, chat_name: str):
    """Delete a chat."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                DELETE FROM user_chats
                WHERE email = %s AND chat_name = %s
            """, (email, chat_name))
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to delete chat: {e}")
        return False


def save_user_profile_to_db(email: str, profile_data: dict):
    """Save user profile."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO user_profiles (
                    email, startup_name, stage, currency, industry, country, founded_date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (email) DO UPDATE SET
                    startup_name = EXCLUDED.startup_name,
                    stage = EXCLUDED.stage,
                    currency = EXCLUDED.currency,
                    industry = EXCLUDED.industry,
                    country = EXCLUDED.country,
                    founded_date = EXCLUDED.founded_date,
                    updated_at = NOW()
            """, (
                email,
                profile_data.get('name'),
                profile_data.get('stage'),
                profile_data.get('currency', 'USD'),
                profile_data.get('industry'),
                profile_data.get('country'),
                profile_data.get('founded_date')
            ))
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Failed to save profile: {e}")
        return False


def get_user_profile_from_db(email: str) -> dict:
    """Get user profile."""
    try:
        conn = get_connection()
        with conn.cursor() as cur:
            cur.execute("""
                SELECT startup_name, stage, currency, industry, country, founded_date
                FROM user_profiles
                WHERE email = %s
            """, (email,))
            row = cur.fetchone()
        conn.close()
        
        if row:
            return {
                "name": row.get('startup_name'),
                "stage": row.get('stage'),
                "currency": row.get('currency', 'USD'),
                "industry": row.get('industry'),
                "country": row.get('country'),
                "founded_date": row.get('founded_date').isoformat() if row.get('founded_date') else None,
            }
        return None
    except Exception as e:
        print(f"❌ Failed to get profile: {e}")
        return None