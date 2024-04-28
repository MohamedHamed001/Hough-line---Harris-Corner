
# Hough Line and Harris Corner Detection

## Project Overview
This project integrates two fundamental image processing techniques: Hough Line Detection and Harris Corner Detection. Implemented in Python using my functions and PyQt for the graphical user interface, this application allows users to interactively upload images, adjust detection parameters, and visualize the detection results for both lines and corners.

## Features
- **Interactive GUI**: A user-friendly interface built with PyQt5, featuring tabs for Hough Line Detection and Harris Corner Detection.
- **Adjustable Detection Parameters**: Users can fine-tune detection parameters like resolution, threshold, and kernel size for tailored detection results.
- **Real-Time Image Processing**: Instantly see the effects of different settings on the processed output, directly within the GUI.

## Technologies Used
- Python 3.x
- OpenCV
- PyQt5
- NumPy
- PIL (Python Imaging Library)

## Installation
First, ensure that Python 3.x is installed on your system. You can then install all required Python libraries using pip. Run the following command in your terminal:

```bash
pip install numpy opencv-python-headless pyqt5 pillow
```

## Running the Application
To launch the application, navigate to the directory containing the script and execute:

```bash
python interface.py
```

## Usage Instructions
1. **Start the Application**: Run the `interface.py` script to open the GUI.
2. **Upload an Image**: Use the 'Browse' button on the respective tab to load an image for processing.
3. **Set Detection Parameters**: Input the desired parameters for either Hough Line or Harris Corner detection.
4. **Apply Detection**: Click 'Apply' to process the image with the current settings and view the results on the screen.

## Application Structure
- `functions.py`: Contains all the core image processing functions, including Gaussian blurring, Sobel filtering, non-maximum suppression, double thresholding, and the main functions for Hough Line and Harris Corner detection.
- `interface.py`: Manages the GUI components, interactions, and integrates image processing functions from `functions.py` to display results.

## Contributing
Contributions to this project are welcome. Please fork the repository, make your changes, and submit a pull request.

## License
This project is licensed under the MIT License - see the [LICENSE.md](LICENSE) file for details.

## Authors
- Mohamed Hamed
##

- `Results and Parameter explanation to be added soon`