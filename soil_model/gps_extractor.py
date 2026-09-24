import pytesseract
from PIL import Image
import re


pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


def extract_gps_data(image_path):

    image = Image.open(image_path)

    text = pytesseract.image_to_string(image)

    # Extract latitude
    lat_match = re.search(r'Lat\s*([0-9.]+)', text)

    # Extract longitude
    lon_match = re.search(r'Long\s*([0-9.]+)', text)

    latitude = (
        lat_match.group(1)
        if lat_match else "Not Found"
    )

    longitude = (
        lon_match.group(1)
        if lon_match else "Not Found"
    )

    # Extract probable location
    lines = text.split("\n")

    location = "Unknown"

    for line in lines:

        if "Andhra" in line or "India" in line:

            location = line.strip()

            break

    # OCR cleanup
    location = (
        location
        .replace("y,", "")
        .replace(".", " ")
        .replace("  ", " ")
        .strip()
    )

    return {

        "text": text,

        "location": location,

        "latitude": latitude,

        "longitude": longitude
    }