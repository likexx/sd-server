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
Imagine a full-color anime-style illustration that captures an intense moment of impact. Extreme Long Shot. From a distance, we see a male super saiya fighter with golden long hair at the center of a powerful explosion, with debris and energy particles flying outwards. The character is in a dynamic pose, possibly crouched or bracing against the shockwave. The explosion radiates a bright spectrum of colors, with the center being the brightest, featuring whites and yellows, and the outer edges transitioning into fiery reds and oranges, then cool blues and purples to convey the immense energy. The character, although smaller in the frame due to the distant view, is detailed with vibrant colors that contrast against the explosion, with their hair and clothes affected by the blast force. The background should depict a battle-worn landscape, further enhancing the scale of the explosion and the distance of the viewpoint.
''',
    style='cartoon2', 
    steps=50,
    numImages=8,
    seed=12,
    deviceType='cuda')
params.image = img_util.convert_image_to_base64('./input/p16.png')

workflow = aigc.AigcWorkflow(params)

images = workflow.generate()
i=31001
for img in images:
    imgBase64Data = img['base64_data']
    processedImage = img_util.add_watermark_to_base64(imgBase64Data, 'created by comicx.ai')
    d = img_util.image_to_base64(processedImage)    
    img_util.saveBase64toPNG(d, './output/{}.png'.format(i))
    i+=1
