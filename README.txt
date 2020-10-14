All of the code that I've written for this project is contained within the main.py file. Each function has a brief description of its parameters and how to call it. 

DEFINING CORRESPONDENCES
For this section, I used plt.ginput() to obtain the corresponding points for two images. The only relevant function for this is get_points.

COMPUTING THE MID-WAY FACE
For this section, I computed the mid-way shape by averaging the labelled points of image1 and image2. With the mid-way shape, I was able to calculate a triangulation of the points, and map both images into this shape. Once both images were mapped, I could compute an average of the pixel value of both images for the midway face. I've erased the code for this section, mainly because frame 20 of my morph_between_two_photos function gives the result. For this section, it is important to look at the transform_face function. It takes in, for example, image1, the labelled points of image1, the midway shape, and the triangulation. It loops through each triangle and inverse maps corresponding pixels over to the new shape as per lecture. 

THE MORPH SEQUENCE
Once I've written transform_image, morphing the sequence was simply a matter of incorporating the warp_frac and dissolve_frac into my design. I.e., instead of simply averageing shapes and pixels, it is now a weighted average. The relevant functioino is morph()

The animated sequence can be found from the function morph_between_two_photos() which takes in strings for picture names as parameters. Depending on whether or not I've already labelled points, I comment/uncomment the function call to get_points. Since most of my morphs had >50 points, I would usually never call get_points. 

THE MEAN FACE OF A POPULATION
I initially started this sectiion by averaging over the danish face set, but I realized soon that if I wanted more interesting results, I would have to find another dataset. I found a free labelled face set from faceresearch.org that categorizes its images into subcategories. To make sure averages are made up of >50 images, I only looked at two subcategories: male and female. The majority of code written for this section is specific to the dataset: as in parsing and computing averages from the data directly. The relevant function in this section would be mean_groups_face(), which directly takes in a group, and computes the average face for this group.

CARICATURES: EXTRAPOLATING FROM THE MEAN
This section was largely an extension of the previous section. The relevant function is caricature(), which takes in my face as an image, and its labelled points. I compute a difference vector in terms of appearance and shape. I extrapolate from my face and morph my face into the extrapolation.

BELLS AND WHISTLES:
For this section, I first changed my gender by computing the difference vector between the average female face and male face. I then extrapolated by adding that vector to my own face. I also made myself even more manly. The relevant function is make_female().

I also tried to compute caricatures and other transformatiions in the new basis. 