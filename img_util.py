from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64

def base64_to_rgb_image(base64_data):
    # Decode the base64 data
    decoded_data = base64.b64decode(base64_data)
    
    # Convert the decoded data to an image
    img_buffer = BytesIO(decoded_data)
    img = Image.open(img_buffer)
    
    # Convert to RGB
    rgb_img = img.convert('RGB')
    
    return rgb_img


def add_watermark(input_image, watermark_text):
    # Make a copy of the input image to ensure original isn't altered
    image = input_image.copy()
    # Get image size
    width, height = image.size

    textSize = width/30

    # Prepare to draw the watermark with default font
    transparent = Image.new('RGBA', image.size, (255, 255, 255, 0))
    d = ImageDraw.Draw(transparent)
    font = ImageFont.truetype("Arial.ttf", textSize)

    # Position the watermark
    text_width, text_height = textSize, textSize
    x = 10  # 10 pixels padding
    y = height - text_height - 10

    # Draw the watermark using an intermediate image to scale the text
    d.text((x, y), watermark_text, fill=(255, 255, 255, 255), font=font, stroke_fill=(3,3,3,128), stroke_width=1)
    watermarked = Image.alpha_composite(image.convert('RGBA'), transparent)

    return watermarked.convert('RGB')
