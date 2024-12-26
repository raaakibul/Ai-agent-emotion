from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt

# Path to the input image
image_path = "img1.jpg"

# Step 1: Load and Display the Image
def load_and_display_image(image_path):
    # Read the image
    img = cv2.imread(image_path)
    # Convert from BGR (OpenCV format) to RGB (matplotlib format)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Display the image
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title("Input Image")
    plt.show()
    return img_rgb

# Step 2: Perform Emotion Recognition
def detect_emotion(image_path):
    try:
        # Analyze emotions using DeepFace
        result = DeepFace.analyze(img_path=image_path, actions=['emotion'])
        dominant_emotion = result['dominant_emotion']
        all_emotions = result['emotion']
        
        print("\nEmotion Analysis:")
        print(f"Dominant Emotion: {dominant_emotion}")
        print("Emotion Scores:")
        for emotion, score in all_emotions.items():
            print(f"{emotion.capitalize()}: {score:.2f}%")
        
        return dominant_emotion
    except Exception as e:
        print(f"Error: {e}")
        return None

# Main function
def main():
    # Load and display the image
    print("Displaying input image...")
    load_and_display_image(image_path)
    
    # Perform emotion detection
    print("Performing emotion recognition...")
    detect_emotion(image_path)

# Execute the program
if __name__ == "__main__":
    main()
