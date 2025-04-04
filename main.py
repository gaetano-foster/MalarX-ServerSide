import socket
import os
import struct

HOST = "0.0.0.0"
PORT = 12345

def predict_image(model, image_path, target_size=(64, 64)):
    class_labels = {0: "Uninfected", 1: "Falciparum", 2: "Vivax"}

    import cv2
    from PIL import Image
    import numpy as np

    img = cv2.imread(image_path)
    if img is None:
        raise ValueError("Could not read image at path: " + image_path)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img).resize(target_size)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, 64, 64, 3)

    predictions = model.predict(img_array)
    class_index = np.argmax(predictions[0])
    confidence = predictions[0][class_index]

    return class_labels[class_index], confidence

def process_image(image_path):
    from keras.models import load_model

    model = load_model("malaria_model.h5")
    predicted_class, confidence = predict_image(model, image_path)

    return f"Prediction: {predicted_class} ({confidence * 100:.2f}% confidence)"


def start_server():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
        server_socket.bind((HOST, PORT))
        server_socket.listen(5)
        print(f"Server listening on {HOST}:{PORT}")

        while True:
            conn, addr = server_socket.accept()
            print(f"Connection from {addr}")

            try:
                # Read file size
                file_size = struct.unpack("!I", conn.recv(4))[0]

                # Read the image file data
                image_data = b""
                while len(image_data) < file_size:
                    packet = conn.recv(file_size - len(image_data))
                    if not packet:
                        break
                    image_data += packet

                # Save image temporarily
                image_path = "received_image.png"
                with open(image_path, "wb") as f:
                    f.write(image_data)

                print(f"Received and saved image: {image_path}")

                # Process the image
                result = process_image(image_path)

                # Send result back to client
                conn.sendall(result.encode())

                # Cleanup
                os.remove(image_path)

            except Exception as e:
                print("Error:", e)
                conn.sendall(b"Error")
            finally:
                conn.close()


if __name__ == "__main__":
    start_server()