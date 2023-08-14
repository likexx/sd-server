import base64, os
from google.cloud import storage

# Initialize a storage client
api_key_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

client = storage.Client()

# Your bucket name and the destination blob name
bucket_name = 'YOUR_BUCKET_NAME'
destination_blob_name = 'YOUR_IMAGE_NAME.jpg'

# Write image to GCS bucket
def upload_to_bucket(local_path, bucket_name, destination_blob_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_path)
    print(f'File {local_path} uploaded to {destination_blob_name}.')

# Read image from GCS bucket and convert to Base64
def read_and_convert_to_base64(bucket_name, blob_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download the image content
    image_content = blob.download_as_bytes()
    
    # Convert the image to Base64
    base64_encoded = base64.b64encode(image_content).decode("utf-8")
    
    return base64_encoded

# Assuming you have an image named 'sample.jpg' in your current directory
upload_to_bucket('sample.jpg', bucket_name, destination_blob_name)

# Read the uploaded image and get its Base64 representation
base64_str = read_and_convert_to_base64(bucket_name, destination_blob_name)
print(base64_str)
