from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reward_code(prompt):
    response = client.chat.completions.create(
        model="gpt-4o-mini",  # Fixed: was "gpt-5-mini" which doesn't exist
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content