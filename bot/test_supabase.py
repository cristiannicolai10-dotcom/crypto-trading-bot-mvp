import os
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")

print("URL:", url)

if not url or not key:
    print("Missing Supabase credentials")
    exit()

supabase = create_client(url, key)

print("Supabase connection successful!")
