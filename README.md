# 🔧 Lot Estimation using OCR and Deep Learning

### Data Curation: Omar
### Programmer: Trevon

## Project Overview

This project combines Optical Character Recognition (OCR) with deep learning and a geometric packing algorithm to estimate the number of potential buildable lots within a given plot boundary. The system processes survey plat images to extract dimensions, calculates theoretical lot placements, and uses a Convolutional Neural Network (CNN) to classify image characteristics, ultimately providing a comprehensive lot prediction.

- **Week 1** - Get dataset, set up environment
- **Week 2** - Train or fine-tune model
- **Week 3** - Test and improve
- **Week 4** - Create demo or video
- **Week 5** - Final testing, documentation
- **Week 6** - Present project

## Features

*   **Multimodal OCR Integration:** Utilizes a Gradio-based OCR model (`prithivMLmods/Multimodal-OCR3`) to accurately extract North-South and East-West boundary measurements from survey plat images.
*   **Rectangle Packing Algorithm:** Employs the `rectpack` library to calculate the maximum number of standard-sized rectangular lots (e.g., 64 ft x 20 ft) that can fit within the extracted plot boundaries.
*   **Custom Convolutional Neural Network (CNN):** A PyTorch-based CNN (`LotPredictor`) is trained to classify images into predefined categories (e.g., 'Eighty', 'Fifty', 'Fourty', 'Ninety', 'Ten', 'Thirty', 'Twenty'), which could correspond to different lot configurations, densities, or other visual characteristics.
*   **Combined Lot Prediction:** Integrates the results from the geometric packing algorithm and the CNN's image classification to generate a final, more nuanced lot estimation.

## Setup and Installation

To set up and run this project, you will need a Python environment and the following libraries. This project was developed in Google Colab, so some steps might be specific to that environment (e.g., Google Drive mounting).

1.  **Clone the repository (if applicable) or open in Google Colab.**
2.  **Install required libraries:**

    ```bash
    !pip install gradio_client langchain_openai langchain langchain_community rectpack torch torchvision torchsummary matplotlib numpy Pillow
    ```

3.  **Google Drive Integration:**
    Mount your Google Drive to access your dataset and image files:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```

4. **Dataset:**
   - Size: 100 Samples
   - Source: Omar's Employer
     

## Usage

1.  **Provide an Image:** Place your survey plat image (e.g., `Image 15-11-2025 at 1.04 PM.png`) in your Colab environment or Google Drive.
2.  **Extract Dimensions:** The Gradio client will process the image and extract the East-West and North-South measurements.

    ```python
    from gradio_client import Client, handle_file

    client = Client("prithivMLmods/Multimodal-OCR3")
    result = client.predict(
    	model_name="Chandra-OCR",
    	text="what are the ft measurements that run east-west and north-south of this plot boundary?",
    	image=handle_file('/content/Image 15-11-2025 at 1.04\u202fPM.png'),
    	max_new_tokens=2048,
    	temperature=0.7,
    	top_p=0.9,
    	top_k=50,
    	repetition_penalty=1.1,
    	api_name="/generate_image"
    )
    # The extracted dimensions will be parsed from `result`
    ```

3.  **Perform Lot Packing:** The `pack_rectangles` function calculates how many predefined lots (e.g., 64x20 ft) can fit.

    ```python
    # Example with extracted plat_width and plat_length
    # total_lots = pack_rectangles(plat_w=float(plat_width), plat_h=float(plat_length), lot_w=64, lot_h=20)
    # print("Total lots from packing:", total_lots["count"])
    ```

4.  **Train/Evaluate CNN:** The `LotPredictor` model is trained and evaluated on your image dataset to learn visual features related to lot characteristics.

5.  **Get Final Lot Prediction:** The system combines the results from lot packing and the CNN's image classification to give a final estimated number of lots.

## Model Details (LotPredictor CNN)

### Architecture

The `LotPredictor` is a custom Convolutional Neural Network implemented in PyTorch, designed for image classification. It consists of multiple convolutional blocks followed by fully connected layers:

*   **Feature Extractor:** A sequential block of `Conv2d`, `ReLU`, and `MaxPool2d` layers, progressively increasing feature map depth (32 -> 64 -> 128 -> 256 -> 512 -> 1024 channels) while reducing spatial dimensions.
*   **Classifier:** A fully connected head with `Linear` layers, `ReLU` activation, and `Dropout` for regularization, mapping the flattened features to the final classification output.

### Input and Output

*   **Input:** 3-channel RGB images of size 512x512 pixels.
*   **Output:** Logits for 7 classes, representing different lot categories: 'Eighty', 'Fifty', 'Fourty', 'Ninety', 'Ten', 'Thirty', 'Twenty'.

### Training Parameters

*   **Loss Function:** `nn.CrossEntropyLoss`
*   **Optimizer:** `Adam` with a learning rate of 0.001.
*   **Epochs:** 50
*   **Batch Size:** 1 (adjusted due to GPU memory constraints)
*   **Transformations:** Images are resized, randomly horizontally flipped, randomly rotated, converted to tensors, and normalized.

## Results

*   **CNN Validation Accuracy:** Achieved approximately 40% accuracy on the validation set after 50 epochs.
*   **Example Lot Packing:** For a plot of 50 ft (East-West) by 127.5 ft (North-South) and standard lots of 64 ft x 20 ft, the `rectpack` algorithm yielded 2 lots.
*   **Combined Prediction:** The final lot prediction integrates both the geometric packing count and the CNN's classification. For instance, an example run resulted in a `Final Lot Prediction: 3.5`.

## Future Work

*   **Improve CNN Performance:** Explore more advanced architectures (e.g., transfer learning with pre-trained models), data augmentation techniques, and hyperparameter tuning to boost classification accuracy.
*   **More Data:** A major reason the CNN classifier has a low accuracy is because of a unequal distribution of training samples. This made it hard for the model to learn meaningful features pertaining to less represented classes.
*   **Refine Lot Packing Logic:** Implement more sophisticated packing strategies to handle irregular plot shapes or varying lot sizes.
*   **Weighted Combination of Results:** Develop a more robust method to combine the OCR-extracted dimensions, geometric packing count, and CNN classification confidence into a more accurate and interpretable final lot prediction.
*   **User Interface:** Create a more interactive interface for uploading images and viewing results.
