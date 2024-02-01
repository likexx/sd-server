# import img_util

# result = img_util.generate_with_dalle3("an asian office lady wearing white shirts and black tight short dress, 20 years old, slim body, sitting on a chair in office, holding a book. Use seed 1234512 to generate the image.", 1)
# print(result)
from openai import OpenAI
client = OpenAI()

setup_prompt = '''
You are an assistant to generate prompts to be used in stable diffusion for image generating. The user will input a line of text in chinese to decribe a story scene, and you should create English prompts to be used in Stable Diffusion to generate the scene as an image. You should also integrate the description for corresponding characters mentiond in the text. The description should be in English.
Here are the chinese introductions for different people. 
黄蓉，中国古代宋朝美女侠客，身兼桃花岛及丐帮等门派绝学。为金庸小说笔下最俏美聪慧的人物之一，黄蓉之美貌冠绝天下且聪明绝顶，足智多谋，心思细密。
吕文德，中国古代襄阳守备官员，中年男人，肥胖且好色。
'''

response = client.chat.completions.create(
  model="gpt-4",
  messages=[
    {"role": "system", "content": setup_prompt},
    {"role": "user", "content": "黄蓉起先就已有提防，此刻只觉吕文德腰部一用力，知道必须马上脱身，便立刻运起当年学自瑛姑的“泥鳅功”，企图从吕文德那肥胖身子重压下脱逃出来。"}
  ]
)
result = response.choices[0].message.content
print(result)