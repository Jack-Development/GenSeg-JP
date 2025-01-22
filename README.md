# GenSeg-JP

GenSeg-JP is a robust and efficient tool for performing OCR (Optical Character Recognition) and advanced segmentation of text (with a focus on Japanese text). This repository demonstrates how to read images, identify characters using EasyOCR, segment them using specialized algorithms, and organize the extracted data in a convenient structure. 

The initial implementation focuses on splitting and segmenting Japanese text from images, but the modular architecture allows for potential expansion to handle various forms of text recognition and processing tasks.

## Skills and Technologies Used

- **Python**  
- **OpenCV**  
- **NumPy**  
- **Matplotlib**  
- **scikit-image**
- **EasyOCR**    

<div>
  <code><img height="50" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/python/python-original.svg" alt="python" /></code>
  <code><img height="50" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/opencv/opencv-original.svg" alt="opencv" /></code>
  <code><img height="50" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/numpy/numpy-original.svg" alt="numpy" /></code>
  <code><img height="50" src="https://raw.githubusercontent.com/devicons/devicon/master/icons/matplotlib/matplotlib-original.svg" alt="matplotlib" /></code>
  <code><img height="50" src="https://avatars.githubusercontent.com/u/897180?v=4" alt="scikit-image" /></code>
</div>

## Getting Started

Below are the essential steps and repository clones to get the project up and running:

1. **Clone the DocRes Repository**  
   ```bash
   cd DocRes
   git clone git@github.com:ZZZHANG-jx/DocRes.git
   ```
2. **Clone the EasyOCR Repository**  
   ```bash
   cd EasyOCR
   git clone git@github.com:JaidedAI/EasyOCR.git
   ```
3. **Set up a Python Virtual Environment and Install Dependencies**  
   ```bash
   python -m venv JapanesePrintWrite
   ./JapanesePrintWrite/bin/activate
   pip install -r requirements.txt
   ```

A comprehensive guide on additional usage, customizing the segmentation pipeline, and detailed explanations of each module will be provided soon.

## Contributing

Contributions, issues, and feature requests are welcome. If you are interested in enhancing the capabilities of **GenSeg-JP** or have discovered any bugs, please open an issue in this repository.

## License

This project is licensed under [CC BY-SA](https://creativecommons.org/licenses/by-sa/4.0/), which allows you to share and adapt the material as long as you provide appropriate credit, link to the license, and share any modifications under the same terms. 

**TLDR:** You are free to use, modify, and distribute the work, but any derivative works must also be shared under the same license.

### Project Structure

Below is a brief overview of the key files in this repository:

- **`start.py`**  
  Main entry point. Generates unique job numbers, reads the input image, runs OCR, and archives output (including inpainted base images, raw cropped images, and cleaned binarized images).

- **`OHTR.py`**  
  Contains advanced segmentation and text recognition algorithms. Uses morphological operations, skeletonization, Voronoi diagrams, and other image processing techniques to split text accurately.

This workflow can be extended or modified to suit various text recognition and segmentation tasks, especially those focusing on printed Japanese text.
