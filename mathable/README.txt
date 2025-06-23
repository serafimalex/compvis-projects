LIBRARY REQUIREMENTS:
matplotlib==3.6.3
numpy==1.24.1
opencv_python==4.7.0.68
Pillow==10.3.0
protobuf==5.26.1
scikit_learn==1.4.2
torch==2.0.0
torchvision==0.15.1
tqdm==4.66.1

HOW TO RUN:
1) Run all cells of Mathable.ipynb
2) Input tests are expected in the 'test' folder. If file structure differs, change value of 'test_path' in the last cell.
3) Output is written to 'submission_files/407_Serafim_Alex'. If this should differ, change value of 'results_path' in the last cell.
4) Overall runtime is expected around 16 minutes. Image preprocessing of the training images should take approx. 15 minutes.
5) Running requires the existence of training images under the same folder structure as the initially provided folder.