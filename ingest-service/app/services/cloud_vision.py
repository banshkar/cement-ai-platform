from google.cloud import vision

client = vision.ImageAnnotatorClient()

def analyze_image(image_path: str):
    with open(image_path, "rb") as f:
        content = f.read()
    image = vision.Image(content=content)
    response = client.label_detection(image=image)
    labels = [label.description for label in response.label_annotations]
    return labels
