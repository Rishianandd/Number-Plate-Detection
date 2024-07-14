import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

# Set path to Tesseract executable (modify the path if needed)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 200)
    return edges

def find_number_plate_contours(edges):
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    number_plate_contours = []
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = w / float(h)
            if 2 < aspect_ratio < 5:
                number_plate_contours.append(approx)
    return number_plate_contours

def extract_number_plate(image, contour):
    x, y, w, h = cv2.boundingRect(contour)
    number_plate = image[y:y+h, x:x+w]
    return number_plate

def display_image(title, image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

def main(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image from {image_path}")
        return
    
    # Preprocess the image
    edges = preprocess_image(image)
    display_image('Edges', edges)
    
    # Find number plate contours
    number_plate_contours = find_number_plate_contours(edges)
    
    # Draw contours on the original image
    contour_image = image.copy()
    cv2.drawContours(contour_image, number_plate_contours, -1, (0, 255, 0), 2)
    display_image('Detected Contours', contour_image)
    
    if number_plate_contours:
        # Extract and display the number plate
        number_plate = extract_number_plate(image, number_plate_contours[0])
        display_image('Number Plate', number_plate)
        
        # OCR to read the text on the number plate
        gray_plate = cv2.cvtColor(number_plate, cv2.COLOR_BGR2GRAY)
        _, thresh_plate = cv2.threshold(gray_plate, 128, 255, cv2.THRESH_BINARY)
        text = pytesseract.image_to_string(thresh_plate, config='--psm 8')
        print("Detected Number Plate Text:", text)
    else:
        print("Number plate not detected.")

if __name__ == "__main__":
    image_path = 'd:/Desktop/DIP/dataset/dip2/car.jpg'  # Update this path
    main(image_path)
