from openai import OpenAI

client = OpenAI(
  base_url = "https://integrate.api.nvidia.com/v1",
  api_key = "$API_KEY_REQUIRED_IF_EXECUTING_OUTSIDE_NGC"
)

completion = client.chat.completions.create(
  model="nvidia/llama-3.1-nemotron-ultra-253b-v1",
  messages=[{"role":"system","content":"detailed thinking on"},{"role":"user","content":"tell me about your self"}],
  temperature=0.6,
  top_p=0.95,
  max_tokens=4096,
  frequency_penalty=0,
  presence_penalty=0,
  stream=True
)

for chunk in completion:
  if chunk.choices[0].delta.content is not None:
    print(chunk.choices[0].delta.content, end="")

