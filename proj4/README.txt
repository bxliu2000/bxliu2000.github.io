The way this directory is organized:

main.py
pt2.py
dataset.py
transforms.py

main.py contains all the code needed to run any of the sections of the project. Inside main.py there exists the 
neural network class corresponding to the one I used for nose detection in part 1. There is also a function called
pt1, which trains the neurala network then runs validation on the testing data. It saves the predictions to a folder.

In pt2.py there is the exact same structure applied to the second part of the project. The parts were quite similar so the
code looks fairly similar. However, there are some minor tweaks. I've imprted pt2 as a function into main so that you can 
call it as well.

In dataset.py, I defined a general dataset class that I use for all three parts of this project. The parameters that the class
takes in are written in the docstrings of the class. Essentially, the only difference between the three parts of the projects
was the size of the images, so that is passed in as a field. This file also contains code to sample some batches to ensure that our
dataset and dataloader are functioning correctly.

In transforms.py, I contain all the code that I used to augment the dataset in part 2. There you can see that I choose to
randomly adjust the saturatiion, brightness, rotation, and lastly I take a random crop of a fixed size. 