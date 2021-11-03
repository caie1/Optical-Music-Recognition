# Assignment 1 Report

## Team Members
- Cody Harris 
- Emma Cai 
- Neelan Scheumann 
## How to Run the Code
To run the program, one simply needs to run the following command:
- python3 omr.py \<filepath to image of sheet music>

## How the program works
The program begins by opening the image and saving an RGB copy to be used later on in the process. Next, the black and white image is passed to the hough transform function. This function returns the y coordinates of the first line from the detected set of five lines as well as distances between the detected lines. Then the three template images are read in and subsequently resized (if necessary) so that they are on the same scale as the staff lines. The program then takes each template, one at a time, and scans the input image to compare that template to each similary sized group of pixels. If the template and the section of the input image are similar enough the coordinate of the upper left hand corner as well as the pitch of the note is stored in a dictionary corrsponding to that template. Finally, one last function takes the coordinates that were just found and draws bounding boxes around each deteceted musical element in the RGB version of the image. The new iamge and a text file detailing what was detected are then saved in the folder. 

## Design choices and Assumptions

### Hough Transform
There were sevearl iterations of the Hough transform function. The first was the standard line detection method that simply found straight lines in the image. This first version was important for understanding the basic idea of the hough transform. For the second iteration, instead of searching for slopes and intercepts in the parameter space, the function looks for the y-coordinate of the staff lines as well as possible distances between each line. This method was much more applicable to this assignment. The third and final version of the function adds in non-maximal supression as well as the ability to automatically detect the number of staves in the image. In v2, the number of staves that needed to be detected was one of the inputs. 

### Converting to a binary iamge
For many of the functions in the program, the input image first needs to be converted into a bnary iamge where the pixel values are either 0 or 1. This makes it easier to perform several of the intermediate steps. In order to accomplish this, a threshold value for what would considered black and what would be considered white needed to be determined. After testing the efficacy of many different threshold values 150 was determined to be optimal. 

### Where to look for notes and rests
Early versions of the program struggled on various input images when looking for rests. Since the eighth rests were nearly all whitespace, many false postiives were being detected. In addition, nearly all of the false positives were in the white space in between the staves where an eighth rest is unlikely to be. Tweaks were made to focus on the staves only when looking for rests. For the notes, the assumption was made the a note will only appear within the staves or at most 4 lines above or below each stave. This was chosen because that was the most extreme case observed in the sample input images. 

### Modifying the template images
As described in the last section the eighth rest template was originally composed of mainly white space. This caused issues with false positives being detected. In addition to focusing on where to look for eighth rests, the template itself was also modified. The top several rows as well as many rows of pixel toward the bottom were cropped out of the template. This resulted in a template that was much more balanced in terms of black and white and this easier to reliably match against. 

### Similarity scores
There were several different similarity scores tested over the course of the assignment. The final one that was used is a mixture of the number of pixels that match between the template and the subsection of the input image and the difference in the proportion of black present in each image. This was done to help make sure that a purely white or black image could successfully have a match above our match threshold. The template and the comparison would also need to have a similar percentage of black and white pixels in order to be a strong match. 

### Skipping columns after detecting an element
After the program detects either a note or a rest, the next few columns in the input image are skipped over when scanning through the image. This is done to help prevent the same note being detected multiple times. While this method helps solve one issue, it does theoretically lead to less than perfect bounding boxes later on. This is becuase the instant a match is detected, another, possibly better match, isn't allowed. If this were a more sophisticated project where location specificity is highly important, then perhaps a better method could be used. However, for this use case, finding the optimal bounding box is likely not necessary and thus a simpler, more time efficient method was utilized. 

## Accuracy
The program performs quite well on test inputs. For music1.png, the program is flawless. It matches every note and rest and correctly assigns the pitch to each note. 

Here are the results for each test image.

| element | metric  | music1 | music2 | music3 | music4 |
| ------- | ------  | ------ | ------ | ------ | ------ |
| note    | correct |  38    |  59    |  208   |  87    |
| note    | total   |  38    |  63    |  217   |  108   |
| e rest  | correct |   4    |   2    |   2    |  17    |
| e rest  | total   |   4    |  12    |  12    |  28    |
| q rest  | correct |   4    |   0    |   0    |   0    |
| q rest  | total   |   4    |   0    |   0    |   0    |

The program performs very well on nearly every image when detecting notes. There were a few times where it returned false negatives after detecting notes within the thick lines connecting several notes. 

When detecting eighth notes, the program struggled. This is especially when attempting to detect two eighth notes that are stacked nearly on top of each other. Becuase this is quite different than the template used it is understandable that this would be difficult to detect. 

Only music1 contained any quarter rests so it is hard to say how well the program would work on detecting quarter rests if the image has some noise. 
