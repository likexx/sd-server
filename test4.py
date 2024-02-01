import aigc
import img_util

params = aigc.AigcParam(
    # prompt="a man is killed by another soldier with a sword, his arm is cut off, his body is cut in half, blood is flooding the ground, view from aside, the killed man lies on the ground and covered with blood, the cut arm is by his body, the soldier is standing aside with a sword with blood", 
    # prompt = "a human body is cut to half but still standing, all blood, the lower half of the body is standing, the upper half body is on the ground",
    # prompt = "a fresh human brain in a simple bag, no human face, no skull, no head, no skull, the brain is taken out from a human's head, the brain has some blood, realistic, photo realistic, details",
    # prompt = "in middle century, in the prison, behind iron fence door, a male prisoner is cruching on the ground, he is a prisoner of war, vivid, colorful, photo realistic",
    # prompt = "Tradition Chinese Ink Painting style of cute, sexsy, green, spring, girl, red belt,watercolor ink painting,full body,Bottom view,film lighting, --ar 9:16 --v 6 --style raw --tile --video --upbeta --uplight --sameseed --stop 20 --chaos 23 --s 4000 --iw 0.7 --quality 1",
#     prompt = '''
# Create a vibrant, full-color anime-style illustration that captures an epic moment of a battle scene. The image should feature two male characters engaged in a powerful showdown. The first character is a super saiya warrior and he is launching a spinning move with a torrent of swirling energy emanating from their fist, their hair and clothing whipped up by the force of their attack. This energy creates a dynamic spiral pattern that dominates the battlefield. The second character is on the defensive, skillfully dodging the attack, their expression one of focused concentration. Bright, vivid colors should highlight the energy and motion, with electric blues and fiery oranges to suggest a high-energy impact. The background should be a blurred whirl of colors that give a sense of rapid movement, with sharp speed lines that emphasize the swift action. The scene is set outdoors, with hints of a grassy field and a clear sky that contrast with the intense action at the forefront.
# ''',
    prompt= '''
Create a vibrant, full-color anime-style illustration of a powerful super saiya fighter experiencing a moment of intense transformation or energy release. The super saiya fighter is in the center of the composition, with a close-up on their face showing a serene yet focused expression. Their eyes are closed, suggesting an inward focus of power. The character's hair is styled in large, upward-spiking tufts that resemble flames and should be colored with a gradient of fiery shades. Energy lines and light rays emanate from the character, radiating outwards in a burst of brilliant light. The energy effect should include a spectrum of bright colors like electric blue, intense yellow, and white at the center, signifying a powerful surge of energy. The character's skin should glow with a soft light, contrasting with the bright background, illustrating the epicenter of the power release.
''',
    style='cartoon', 
    steps=50,
    numImages=4,
    seed=12,
    deviceType='cuda')
params.image = img_util.convert_image_to_base64('./input/p8.png')

client = aigc.AigcWorkflow(params)

images = client.generate()
i=30001
for img in images:
    imgBase64Data = img['base64_data']
    processedImage = img_util.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
    d = img_util.image_to_base64(processedImage)    
    img_util.saveBase64toPNG(d, './output/{}.png'.format(i))
    i+=1
