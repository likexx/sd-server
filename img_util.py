from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import base64, os
import base64
import requests
from openai import OpenAI

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


def generate_with_dalle3(prompt, k):
    # export OPENAI_API_KEY='your-api-key-here'
    apiKey = os.getenv("OPENAI_API_KEY")
    if apiKey != "":
        print("found api key")
    client = OpenAI()

    result = []

    while k > 0:
        response = client.images.generate(
            model="dall-e-3",
            prompt=prompt,
            size="1024x1024",
            quality="standard",
            n=1,
        )
        imageUrl = response.data[0].url
        base64data = image_url_to_base64(imageUrl)
        result.append[{"base64_str": base64data}]
        k-=1
    
    return result

# Function to convert an image URL to a base64 string
def image_url_to_base64(url):
    # Send a GET request to the image URL
    response = requests.get(url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Open the image as a file-like object using BytesIO
        image = Image.open(BytesIO(response.content))
        
        # Convert the image to the desired format (e.g., JPEG, PNG)
        buffered = BytesIO()
        image_format = 'PNG'  # or 'JPEG'
        image.save(buffered, format=image_format)
        
        # Encode the image data to base64
        img_str = base64.b64encode(buffered.getvalue())
        
        # If you want the base64 string in text format
        img_str = img_str.decode('utf-8')
        
        return img_str
    else:
        print("Failed to retrieve the image.")
        return None