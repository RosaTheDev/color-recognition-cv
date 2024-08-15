# Color Recognition with Computer Vision

## Inspiration
I wanted to do this project out of inspiration from [@iotlab616](https://www.tiktok.com/@iotlab616) on TikTok. Her projects are amazing, and I recommend others check out her page!

## Motivation
Since this is my first time working with or even hearing about Computer Vision, I decided to embark on this journey with the help of ChatGPT-4. I'll be using it as a guide to bounce my thoughts back and forth, and I plan to make detailed notes on the steps we worked on together.

## Project Goals
- **Goal 1**: Detect and recognize specific colors (e.g., Red, Green, Blue) in real-time using a webcam feed.
- **Goal 2**: Draw bounding boxes around the detected color regions.
- **Goal 3**: Optimize the detection process to reduce false positives and improve accuracy.

## Project Setup
### Prerequisites
Before you begin, ensure you have met the following requirements:
- Python 3.x installed on your local machine
- Installed the following Python libraries:
  - `opencv-python`
  - `Pillow`
  
### Installation
1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/color-recognition-cv.git
   cd color-recognition-cv

2. **Install opencv-python**:
   ```bash
   pip install opencv-python

3. **Install pytest for testing**
   ```bash
   pip install pytest


## Findings, What I Learned
### Startup Findings
- Started with getting the webcam to start up, this is done with ``` cv2.VideoCapture(0)``` the reason that we use 0 as the parameter here is because it is the default webcam
- The reason we are using the HSV (Hue, Saturation, and Value) color space is because it is more effective for color detection than the RGB color space. This makes it easier to detect specific colors under different lighting conditions

### Testing Findings
- Seting up testing for this was interesting I've never used pytest before but also how would one even test a camera detecting colors? Here I hope to break it down to some of the issues I ran into
- Importing the ```'main'``` module, solved by importing sys and importing os which allowed the test file to find the 'main.py' file: 
```bash
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
```
- Typo, when we are using image processing we use ```numpy or np```, ```uint8```, but it can be confused for unit so I make the mistake of type ```unit8``` instead
- Array nesting it can be confusing dealing with nested arrays, and keeping track of where the values are, its been fixed it should look like this: 
```
bgr_image = np.array([
    [[255, 0, 0], [255, 0, 0]], 
    [[255, 0, 0], [255, 0, 0]]
], dtype=np.uint8)
```

### CI/CD Pipelines
This is my first time setting up my own CI/CD pipelines, so bare with me if you view initially there was a lot ran, let me go into detail of what I achieved:
- First issue that I ran into was that the ```requirements.txt``` file was not being recognized
- Second issue was that the pipelines ran into an issue with starting up the webcam, probably because the pipeline itself doesnt have access to a proper webcam on the computer since the pipeline is hosted on github, this issue was fixed by mocking the webcam in the CI enviornment

## Further Optimzation
- Trying to test how much more optimized I could get with using Chat GPT 4o, so far we had manage to add more mocking for the webcam testing.
- We moved our imshow functions into a helper function, this really cleaned up the code a bit more.
- Learned that apparently these -> ```""" """``` are called **Docstrings** and these are used to describe what a function, class or module does, they also can be accessed using tools like Pythons built-in "help()" function or through code introspection, meanwhile (I was previously using these) ```#``` are used for regular comments to eplain specific lines or blocks of code. (basically use the Docstrings for the helper functions and use the use comments for the blocks inside of the main function
- Resizing the frames, we resized the frames to be on a smaller scale then they were originally, this reduces the data being processed leading to faster execution, I personally didn't notice a change in resoultion when implmented, but I did notice that it is much faster to close the frames pressing ```Q``` now Chat GPT  4o mentions that the faster quit time inidicates that the program is handling the frames more efficently so I think we are on the right track.


