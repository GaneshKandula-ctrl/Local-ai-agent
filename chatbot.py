import ollama

client = ollama.Client()

model = 'llama3.2'
prompt = 'What is your name?'

response = client.generate(model=model, prompt=prompt)
print(response.response)