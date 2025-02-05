# Enhancing-Image-resolution-with-Super-Resolution-Gan

I used a facial image dataset from Kaggle to create my own dataset. I extracted 10 patches of 
size 128x128 pixels from each image randomly. This acted as my high-resolution image. To 
ensure that the patches were not of blank backgrounds of the pictures I restricted the random 
number that represent the bottom left corner of the patch to be within the inner half of the 
dimension of the image. The I used down sampling to down sample each 128x128 pixel patch 
to 64x64 pixel patch. This acted as my low resolution image. 

![image](https://github.com/user-attachments/assets/086d6210-47f2-42e4-a6e1-f04fc94c1f0c)


Overall, this project has demonstrated the potential of SRGANs in enhancing the resolution 
of facial images, paving the way for their application in various domains where high-quality 
visual representations are critical. By addressing the limitations and exploring the future 
research directions outlined above, we can further advance the state-of-the-art in facial image 
super-resolution and contribute to the broader field of image enhancement and restoration.
One key problem is the shear volume of dataset it actually required to generate super￾resolution images. For this project, I didn’t possess the GPU power or the dataset volume 
required to perform super-resolution properly. 
