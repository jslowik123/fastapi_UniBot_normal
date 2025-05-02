from openai import OpenAI
client = OpenAI()

completion = client.chat.completions.create(
    model="gpt-4.1",
    messages=[
        {
            "role": "user",
            "content": "Expand this query: 'Zulassungsvoraussetzungen für BWL'"
        }
    ]
)

print(completion.choices[0].message.content)  