import os
import sys
from services.ai_connector import process_image

def main():
    print("Testing pipeline on sample_photo.png (with face)...")
    res_face = process_image("sample_photo.png")
    print(f"Result for sample_photo.png: {res_face['processed_path']} (faces detected: {res_face['faces_detected']})")
    print("--------------------------------------------------")
    print("Testing pipeline on color_test.png (no face)...")
    res_noface = process_image("color_test.png")
    print(f"Result for color_test.png: {res_noface['processed_path']} (faces detected: {res_noface['faces_detected']})")
    print("--------------------------------------------------")
    print("Testing pipeline on audrey_grayscale.jpg (with face)...")
    res_audrey = process_image("audrey_grayscale.jpg")
    print(f"Result for audrey_grayscale.jpg: {res_audrey['processed_path']} (faces detected: {res_audrey['faces_detected']})")
    print("All tests completed successfully!")

if __name__ == "__main__":
    main()
