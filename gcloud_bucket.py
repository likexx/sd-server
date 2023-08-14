import base64, os
from google.cloud import storage

# Initialize a storage client
credential_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
if not credential_path:
    print("cannot find env value: GOOGLE_APPLICATION_CREDENTIALS")
    exit(1)

client = storage.Client()

# Your bucket name and the destination blob name
user_img_bucket_name = 'aigc-user-image'
aigc_img_bucket_name = 'aigc-ai-image'

# Write image to GCS bucket
def upload_to_bucket(base64_data, bucket_name, destination_blob_name):
    """
    Upload base64 encoded image data to Google Cloud Storage.

    :param base64_data: The base64 encoded image data.
    :param bucket_name: Name of the GCS bucket.
    :param destination_blob_name: Desired blob name for the uploaded image.
    """
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    
    # Decode the base64 data
    image_content = base64.b64decode(base64_data)
    
    # Upload the decoded data to GCS
    blob.upload_from_string(image_content)
    print(f'Image uploaded to {destination_blob_name} in bucket {bucket_name}.')


# Read image from GCS bucket and convert to Base64
def read_file_from_bucket(bucket_name, blob_name):
    bucket = client.get_bucket(bucket_name)
    blob = bucket.blob(blob_name)
    
    # Download the image content
    image_content = blob.download_as_bytes()
    
    # Convert the image to Base64
    base64_encoded = base64.b64encode(image_content).decode("utf-8")
    
    return base64_encoded

# # Assuming you have an image named 'sample.jpg' in your current directory
# upload_to_bucket('sample.jpg', bucket_name, destination_blob_name)

# # Read the uploaded image and get its Base64 representation
# base64_str = read_and_convert_to_base64(bucket_name, destination_blob_name)
# print(base64_str)
