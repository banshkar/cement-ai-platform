from google.genai import Client

client = Client(api_key="AIzaSyBt_UZ7K3RMoJ-MSQestGuHaJ6agOkUG-E")

# ----------------------------
# Chat Example
# ----------------------------
chat_response = client.chats.create(
    model="gemini-1.5-flash"
)

print("=== Chat Response ===")
print(chat_response)

# ----------------------------
# Text Generation Example
# ----------------------------
text_response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Write a short story about a robot learning to cook."
)

print("\n=== Generated Text ===")
print(text_response.text)
