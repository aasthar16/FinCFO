# test_db.py
import streamlit as st
from config.database import get_checkpointer, verify_tables, init_tables

print("=" * 50)
print("Testing PostgreSQL with psycopg3")
print("=" * 50)

# Check secrets
try:
    print(f"✅ Postgres URL: {st.secrets['postgres']['url'][:30]}...")
except Exception as e:
    print(f"❌ Error reading secrets: {e}")
    exit(1)

# Verify tables
print("\nChecking tables...")
if verify_tables():
    print("✅ Tables exist")
else:
    print("❌ Tables don't exist. Creating...")
    init_tables()
    print("✅ Tables created")

# Get checkpointer
print("\nGetting checkpointer...")
try:
    checkpointer = get_checkpointer()
    print("✅ Checkpointer created successfully!")
except Exception as e:
    print(f"❌ Error: {e}")