import cv2
import os

images_dir = "./"

annotations_file = "annotations.txt"


def annotate_image(image_path):
    img = cv2.imread(image_path)
    cv2.imshow("Annotate the Grape Bunch (Press 'Enter' to save)", img)
    roi = cv2.selectROI(
        "Annotate the Grape Bunch (Press 'Enter' to save)", img)
    cv2.destroyAllWindows()
    x, y, w, h = roi
    annotation = f"{x} {y} {w} {h}"

    return annotation


with open(annotations_file, 'w') as file:
    for image_file in os.listdir(images_dir):
        if image_file.endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(images_dir, image_file)
            annotation = annotate_image(image_path)
            file.write(f"{image_path} {annotation}\n")

print(f"Annotations file generated: {annotations_file}")
