from openai import OpenAI

client = OpenAI()


instructions = """
ComicX is programmed to generate comic images directly based on user requests, only posing clarification questions when necessary to ensure accuracy. It operates with a professional tone in Chinese and utilizes uploaded PDFs to inform the style and narrative of the comics it helps to create. ComicX streamlines the creative process by providing immediate visual drafts, enriching the storytelling experience for creators.
"""

refImageFile1 = client.files.create(
  file=open("dragonball33.pdf", "rb"),
  purpose='assistants'
)
refImageFile2 = client.files.create(
  file=open("dragonball34.pdf", "rb"),
  purpose='assistants'
)

assistant = client.beta.assistants.create(
    name="Comicx Assistant",
    instructions=instructions,
    model="gpt-4-1106-preview",
    tools=[{"type": "retrieval"}],
    file_ids=[refImageFile1.id, refImageFile2.id],
)

thread = client.beta.threads.create()

print("thread createad")

message = client.beta.threads.messages.create(
    thread_id=thread.id,
    role="user",
    content="create an image: a girl is sitting on a chair, with white short dress"
)

run = client.beta.threads.runs.create(
  thread_id=thread.id,
  assistant_id=assistant.id
)

while run.status != 'completed':
    run = client.beta.threads.runs.retrieve(
    thread_id=thread.id,
    run_id=run.id
    )
    print(run.status)


messages = client.beta.threads.messages.list(
  thread_id=thread.id
)

print(messages)


