<h2>TensorFlow-FlexUNet-Image-Segmentation-Multiclass-Pavement-Defect (2025/11/05)</h2>
<!--
Toshiyuki Arai<br>
Software Laboratory antillia.com<br>
 -->
This is the first experiment of Image Segmentation for <b>Multiclass Pavement Defect</b> based on 
our <a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet</a>
 (<b>TensorFlow Flexible UNet Image Segmentation Model for Multiclass</b>)
, and a 512x512 pixels 
Augmented-Pavement-ImageMask-Dataset 
 which was derived by us from   
<a href="https://github.com/FuturePave/PaveSeg-Dataset">
<b>PaveSeg Dataset</b>
</a>
<br><br>
<b>Data Augmentation Strategy</b><br>
To address the limited size of <b>img_previw</b> and <b>label_preview</b> of the <b>PaveSeg Dataset</b>,
which contains 100 images and labels repectively,

we used our offline augmentation tools <a href="https://github.com/sarah-antillia/Image-Distortion-Tool"> 
Image-Distortion-Tool</a> and
<a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a> 
 to augment the preview dataset.

<br>
<br>
<hr>
<b>Actual Image Segmentation for Images of 512x512 pixels</b><br>
As shown below, the inferred masks predicted by our segmentation model trained on the 
PNG dataset appear similar to the ground truth masks, but this model failed to segment some blue crack regions.<br>
<b>rgb_map (Crack:blue, Pothole:green, Sealed_crack:red, Patch:pink, Alligator_crack:cyan,
 Utilitity_cover:yellow, Expansion_joint:white)</b><br>
<br>
<table>
<tr>
<th>Input: image</th>
<th>Mask (ground_truth)</th>
<th>Prediction: inferred_mask</th>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1002_0.3_0.3_11176.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1002_0.3_0.3_11176.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1002_0.3_0.3_11176.png" width="320" height="auto"></td>
</tr>
</tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1003_0.3_0.3_10253.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1003_0.3_0.3_10253.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1003_0.3_0.3_10253.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1003_0.3_0.3_14656.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1003_0.3_0.3_14656.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1003_0.3_0.3_14656.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>
<h3>1 Dataset Citation</h3>
The dataset used here was obtained from:
<br><br>
<b>img_preview</b> and <b>label_preivew</b> of
<a href="https://github.com/FuturePave/PaveSeg-Dataset">
<b>PaveSeg Dataset</b>
</a>
<br>
<br>
The dataset provides a comprehensive collection of annotated images designed for high-precision pavement 
condition recognition through semantic segmentation. <br>
By focusing on detailed labeling of pavement distresses, it enables the development and evaluation of 
deep learning models aimed at automating the identification and classification of pavement defects. The dataset offers a rich set of high-quality, 
labeled data that supports advancements in automated road monitoring, aiming to pave the way for improved predictive maintenance and
<br><br>


<b>Annotations</b><br>
Each label in <b>label_preview</b> subset is gray-scaled with the following 7 categories.<br> 
<table <table border="1" style="border-collapse: collapse;">
<tr>
<th>Category ID</th><th>Category Name</th><th>Grayscale Value</th>
</tr>
<tr><td>1</td><td>Crack          </td><td>  30</td></tr>
<tr><td>2</td><td>Pothole        </td><td>  60</td></tr>
<tr><td>3</td><td>Sealed crack   </td><td>  90</td></tr>
<tr><td>4</td><td>Patch          </td><td> 120</td></tr>
<tr><td>5</td><td>Alligator crack</td><td> 150</td></tr>
<tr><td>6</td><td>Utilitity_cover</td><td> 180</td></tr>
<tr><td>7</td><td>Expansion_joint</td><td> 210</td></tr>
<tr><td>0</td><td>Background     </td><td>   0</td></tr>
</table>
<br>
<br>
<b>Data Access</b><br>
A preview version of the dataset, containing 100 images and their corresponding annotations, 
is available in this repository. <br>
Anyone can apply for access to the full dataset for non-commercial use by filling out a simple data request form. 
<br>We will provide the application process for obtaining the complete dataset as soon as possible, in preparation for its public release.<br>
 Additionally, we are in the process of collecting and annotating more data to meet the demands of future research.
If you use this dataset for academic research, please cite it using the following reference:
<br>
<b>
[Under Review] Pavement condition sensing for vision-based auton-omous driving based on deep encoder-decoder net-work and spatial attention mechanism
</b>
<br><br>

<b>LICENSE</b><br>
Unknown
<br>
<br>
<h3>
2 Pavement ImageMask Dataset
</h3>
 If you would like to train this Pavement Segmentation model by yourself,
 please generate your own Augmented-Pavement-ImageMask-Dataset by using
 the augmentation tools <a href="https://github.com/sarah-antillia/Image-Distortion-Tool"> 
Image-Distortion-Tool</a> and
<a href="https://github.com/sarah-antillia/Barrel-Image-Distortion-Tool">Barrel-Image-Distortion-Tool</a>,
and put it under <b>dataset</b> folder to be:
<pre>
./dataset
└─Pavement
    ├─test
    │   ├─images
    │   └─masks
    ├─train
    │   ├─images
    │   └─masks
    └─valid
        ├─images
        └─masks
</pre>
<b>Pavement Statistics</b><br>
<img src ="./projects/TensorFlowFlexUNet/Pavement/Pavement_Statistics.png" width="512" height="auto"><br>
<br>
As shown above, the number of images of train and valid datasets is not large to use for a training set of our segmentation model.
<br><br>
<b>Mask Colorization</b><br>
We used the following <b>Grayscale</b> and <b>RGB</b> color mapping table to generate the multiclass colorized mask.
<br>
<table <table border="1" style="border-collapse: collapse;">
<tr>
<th>Index</th>
<th>Category</th><th>Grayscale </th><th>Color</th><th>RGB </th>
</tr>
<tr><td>1</td><td>Crack          </td><td>  30</td><td>blue  </td><td>(  0,  0,255)</td></tr>
<tr><td>2</td><td>Pothole        </td><td>  60</td><td>green </td><td>(  0,255,  0)</td></tr>
<tr><td>3</td><td>Sealed crack   </td><td>  90</td><td>red   </td><td>(255,  0,  0)</td></tr>
<tr><td>4</td><td>Patch          </td><td> 120</td><td>pink  </td><td>(255,  0,255)</td></tr>
<tr><td>5</td><td>Alligator crack</td><td> 150</td><td>cyan  </td><td>(  0,255,255)</td></tr>
<tr><td>6</td><td>Utilitity_cover</td><td> 180</td><td>yellow</td><td>(255,255,  0)</td></tr>
<tr><td>7</td><td>Expansion_joint</td><td> 210</td><td>white </td><td>(255,255,255)</td></tr>
</table>

<br><br>
<b>Train_images_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/train_images_sample.png" width="1024" height="auto">
<br>
<b>Train_masks_sample</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/train_masks_sample.png" width="1024" height="auto">
<br>
<br>
<h3>
3 Train TensorFlowUNet Model
</h3>
 We trained Pavement TensorFlowFlexUNet Model by using the following
<a href="./projects/TensorFlowFlexUNet/Pavement/train_eval_infer.config"> <b>train_eval_infer.config</b></a> file. <br>
Please move to ./projects/TensorFlowFlexUNet/Pavement and, and run the following bat file.<br>
<pre>
>1.train.bat
</pre>
, which simply runs the following command.<br>
<pre>
>python ../../../src/TensorFlowFlexUNetTrainer.py ./train_eval_infer.config
</pre>
<hr>

<b>Model parameters</b><br>
Defined a small <b>base_filters = 16</b> and large <b>base_kernels = (7,7)</b> for the first Conv Layer of Encoder Block of 
<a href="./src/TensorFlowFlexUNet.py">TensorFlowFlexUNet.py</a> 
and a large num_layers (including a bridge between Encoder and Decoder Blocks).
<pre>
[model]
image_width    = 512
image_height   = 512
image_channels = 3

num_classes    = 2

base_filters   = 16
base_kernels   = (7,7)
num_layers     = 8
dropout_rate   = 0.05
dilation       = (1,1)

</pre>

<b>Learning rate</b><br>
Defined a very small learning rate.  
<pre>
[model]
learning_rate  = 0.00007
</pre>

<b>Online augmentation</b><br>
Disabled our online augmentation.  
<pre>
[model]
model         = "TensorFlowFlexUNet"
generator     = False
</pre>

<b>Loss and metrics functions</b><br>
Specified "categorical_crossentropy" and <a href="./src/dice_coef_multiclass.py">"dice_coef_multiclass"</a>.<br>
You may specify other loss and metrics function names.<br>
<pre>
[model]
loss           = "categorical_crossentropy"
metrics        = ["dice_coef_multiclass"]
</pre>
<b>Learning rate reducer callback</b><br>
Enabled learing_rate_reducer callback, and a small reducer_patience.
<pre> 
[train]
learning_rate_reducer = True
reducer_factor     = 0.5
reducer_patience   = 4
</pre>

<b>Early stopping callback</b><br>
Enabled early stopping callback with patience parameter.
<pre>
[train]
patience      = 10
</pre>

<b>RGB Color map</b><br>
rgb color map dict for Pavement 1+1 classes.
<pre>
[mask]
mask_datatype    = "categorized"
mask_file_format = ".png"
;Crack:blue, Pothole:green, Sealed_crack:red, Patch:pink, Alligator_crack:cyan,Utilitity_cover:yellow, Expansion_joint:white
rgb_map = {(0,0,0):0,(0,0,255):1, (0,255,0):2, (255,0,0):3, (255,0,255):4,(0,255,255):5, (255,255,0):6, (255,255,255):7}
</pre>

<b>Epoch change inference callback</b><br>
Enabled <a href="./src/EpochChangeInferencer.py">epoch_change_infer callback (EpochChangeInferencer.py)</a></b>.<br>
<pre>
[train]
epoch_change_infer       = True
epoch_change_infer_dir   =  "./epoch_change_infer"
num_infer_images         = 6
</pre>

By using this callback, on every epoch_change, the inference procedure can be called
 for 6 images in <b>mini_test</b> folder. This will help you confirm how the predicted mask changes 
 at each epoch during your training process.<br> <br> 

<b>Epoch_change_inference output at starting (epoch 1,2,3)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/epoch_change_infer_at_start.png" width="1024" height="auto"><br>
<br>
<b>Epoch_change_inference output at middlepoint (epoch 44,45,46)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/epoch_change_infer_at_middlepoint.png" width="1024" height="auto"><br>
<br>

<b>Epoch_change_inference output at ending (epoch 89,90,91)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/epoch_change_infer_at_end.png" width="1024" height="auto"><br>
<br>


In this experiment, the training process was stopped at epoch 91 by EarlyStopping Callback.<br><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/train_console_output_at_epoch91.png" width="720" height="auto"><br>
<br>

<a href="./projects/TensorFlowFlexUNet/Pavement/eval/train_metrics.csv">train_metrics.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/eval/train_metrics.png" width="520" height="auto"><br>

<br>
<a href="./projects/TensorFlowFlexUNet/Pavement/eval/train_losses.csv">train_losses.csv</a><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/eval/train_losses.png" width="520" height="auto"><br>

<br>

<h3>
4 Evaluation
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Pavement</b> folder,<br>
and run the following bat file to evaluate TensorFlowUNet model for Pavement.<br>
<pre>
./2.evaluate.bat
</pre>
This bat file simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetEvaluator.py ./train_eval_infer.config
</pre>

Evaluation console output:<br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/evaluate_console_output_at_epoch91.png" width="720" height="auto">
<br><br>Image-Segmentation-Pavement

<a href="./projects/TensorFlowFlexUNet/Pavement/evaluation.csv">evaluation.csv</a><br>

The loss (bce_dice_loss) to this Pavement/test was not low, and dice_coef not high as shown below.
<br>
<pre>
categorical_crossentropy,0.0596
dice_coef_multiclass,0.9807
</pre>
<br>
<h3>
5 Inference
</h3>
Please move to a <b>./projects/TensorFlowFlexUNet/Pavement</b> folder<br>
,and run the following bat file to infer segmentation regions for images by the Trained-TensorFlowFlexUNet model for Pavement.<br>
<pre>
./3.infer.bat
</pre>
This simply runs the following command.
<pre>
python ../../../src/TensorFlowFlexUNetInferencer.py ./train_eval_infer.config
</pre>
<hr>
<b>mini_test_images</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/mini_test_images.png" width="1024" height="auto"><br>
<b>mini_test_mask(ground_truth)</b><br>
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/mini_test_masks.png" width="1024" height="auto"><br>

<hr>
<b>Inferred test masks</b><br>
 
<img src="./projects/TensorFlowFlexUNet/Pavement/asset/mini_test_output.png" width="1024" height="auto"><br>
<br>
<hr>
<b>Enlarged images and masks for Images of 512x512 pixels </b><br>
<b>rgb_map (Crack:blue, Pothole:green, Sealed_crack:red, Patch:pink, Alligator_crack:cyan,<br>
 Utilitity_cover:yellow, Expansion_joint:white)</b> <br>

<table>
<tr>

<th>Image</th>
<th>Mask (ground_truth)</th>
<th>Inferred-mask</th>
</tr>

<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1002_0.3_0.3_12701.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1002_0.3_0.3_12701.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1002_0.3_0.3_12701.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1002_0.3_0.3_13081.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1002_0.3_0.3_13081.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1002_0.3_0.3_13081.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1003_0.3_0.3_14465.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1003_0.3_0.3_14465.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1003_0.3_0.3_14465.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1002_0.3_0.3_10946.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1002_0.3_0.3_10946.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1002_0.3_0.3_10946.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/deformed_alpha_1300_sigmoid_8_10496.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/deformed_alpha_1300_sigmoid_8_10496.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/deformed_alpha_1300_sigmoid_8_10496.png" width="320" height="auto"></td>
</tr>
<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/deformed_alpha_1300_sigmoid_8_10703.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/deformed_alpha_1300_sigmoid_8_10703.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/deformed_alpha_1300_sigmoid_8_10703.png" width="320" height="auto"></td>
</tr>

<tr>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/images/barrdistorted_1004_0.3_0.3_13570.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test/masks/barrdistorted_1004_0.3_0.3_13570.png" width="320" height="auto"></td>
<td><img src="./projects/TensorFlowFlexUNet/Pavement/mini_test_output/barrdistorted_1004_0.3_0.3_13570.png" width="320" height="auto"></td>
</tr>
</table>
<hr>
<br>

<h3>
References
</h3>
<b>1. A REVIEW OF PAVEMENT DEFECT DETECTION BASED ON VISUAL PERCEPTION</b><br>
Chenchen Yang, Li Yang, Hailong Duan, Jingwei Deng<br>
<a href="https://ijomam.com/wp-content/uploads/2024/09/pag.-131-146_A-REVIEW-OF-PAVEMENT-DEFECT-DETECTION-BASED-ON-VISUAL-PERCEPTION.pdf">
https://ijomam.com/wp-content/uploads/2024/09/pag.-131-146_A-REVIEW-OF-PAVEMENT-DEFECT-DETECTION-BASED-ON-VISUAL-PERCEPTION.pdf
</a>
<br>
<br>
<b>2. Multi-temporal crack segmentation in concrete structures using deep learning approaches</b><br>
Said Harb, Pedro Achanccaray, Mehdi Maboudi, Markus Gerke<br>
<a href="https://arxiv.org/pdf/2411.04620">https://arxiv.org/pdf/2411.04620</a>
<br>
<br>
<b>3. An efficient semantic segmentation method for road crack based on EGA-UNet</b><br>
Li Yang, Jingwei Deng, Hailong Duan & Chenchen Yang <br>
<a href="https://www.nature.com/articles/s41598-025-01983-3">https://www.nature.com/articles/s41598-025-01983-3<</a>
<br>
<br>
<b>4. EfficientPavementNet: A Lightweight Model for Pavement Segmentation</b><br>
Abid Hasan Zim, Aquib Iqbal, Zaid Al-Huda, Asad Malik, Minoru Kuribayash<br>
<a href="https://arxiv.org/pdf/2409.18099">https://arxiv.org/pdf/2409.18099</a>
<br>
<br>
<b>5. Fine-grained crack segmentation for high-resolution images via a multiscale cascaded network</b><br>
Honghu Chu, Pang-jo Chun<br>
<a href="https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13111">https://onlinelibrary.wiley.com/doi/full/10.1111/mice.13111</a>
<br>
<br>
<b>6. TensorFlow-FlexUNet-Image-Segmentation-Model</b><br>
Toshiyuki Arai antillia.com <br>
<a href="https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model">
https://github.com/sarah-antillia/TensorFlow-FlexUNet-Image-Segmentation-Model
</a>
<br>
<br>

