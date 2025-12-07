# B8-alpha-identification

My 3rd year project for the BA Physics at Oxford University in Hilary Term 2025, which was given a mark of 75%. Both my final report and presentation slides are included in the repository. 

The aim of the model is to automate the counting of alpha particles from the images outputted by a Rutherford scattering experiment.

**Input:** Images which have many white dots on a black background. Alpha particles produce large, brighter spots which need to be distinguished from the smaller, dimmer background noise. Images are taken at a variety of angles from 0° to 160° and the distribution of alpha particles depends strongly on this angle.

**Output:** The estimated count - the number of alphas in the image.

## Model Architecture
The model uses convolutional layers and a residual module to extract features before flattening. Then two fully connected layers produce a single output count. 

## Model Training
A base model is trained using synthetic data, created using real background noise (images with no alpha particles) overlaid with Gaussian spots to simulate alpha hits.

Then the model is finetuned with real data. To increase the dataset size, real images are augmented and turned into many images with the same count through transformations such as rotations and reflections.

At this point the model is split in two, one finetuned for the low-angle images which have many alpha particles, and one for the high-angle images which have very few.

## Results
The models achieved a 98.8% accuracy for low-angle images and 79% accuracy for high-angle images. 

A full discussion of the model, data, training, results and limitations can be found in the report.

## Acknowledgements
A massive thank you to Prof. Todd Huffman for supervising the project, and to Will for collecting so much experimental data and patiently labelling lots of it while I got this to work.
