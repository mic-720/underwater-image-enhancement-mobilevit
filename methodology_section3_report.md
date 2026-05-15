# 3. METHODOLOGIES

## 3.1 Introduction

The methodology chapter explains the technical approach followed in the proposed underwater image enhancement project. In any deep learning-based system, methodology is a crucial part of the report because it describes the logical and scientific process used to solve the identified problem. It presents the theoretical background of the selected algorithms, their suitability for the application, and the way they relate to the overall design of the project. In this work, the methodology is centered on deep learning approaches, with particular emphasis on neural network architectures that are widely used in pattern recognition and image analysis.

The proposed project focuses on underwater image enhancement, which is a computer vision problem involving the restoration of low-quality underwater images. Underwater images often contain severe distortions due to the absorption and scattering of light in water. These distortions may include bluish or greenish colour cast, reduced visibility, blurred edges, haze, low contrast, and non-uniform illumination. Traditional enhancement techniques such as histogram equalization, white balancing, and filtering can improve some visual aspects, but they often fail to adapt to the complex and highly variable nature of underwater degradation. For this reason, deep learning-based approaches have become increasingly important in solving such problems.

Deep learning is well suited to image enhancement because it can automatically learn complex feature representations from large amounts of data. Instead of relying solely on manually designed image processing rules, deep learning systems learn the relationship between degraded images and improved outputs through training. This allows them to capture both local features, such as edges and textures, and global characteristics, such as colour balance and overall scene structure.

The methodologies considered in this chapter include three major deep learning algorithms:

- Convolutional Neural Network (CNN)
- Recurrent Neural Network (RNN)
- Long Short-Term Memory (LSTM)

Among these, CNN is the most relevant to the proposed project because it is specifically designed for image data and spatial feature extraction. RNN and LSTM are also important deep learning architectures and are studied in this chapter because they represent core algorithmic families in modern artificial intelligence. Although they are more suitable for sequential and time-dependent data than static image enhancement, discussing them strengthens the theoretical foundation of the report and shows comparative understanding of neural network methodologies.

This chapter therefore serves two purposes. First, it explains deep learning as the main methodological framework used in the project. Second, it compares CNN, RNN, and LSTM in terms of their design, working mechanism, strengths, limitations, and relevance to underwater image enhancement. The chapter concludes by justifying why CNN-based methodology is the most suitable for the present system.

## 3.2 Deep Learning as the Core Methodological Framework

Deep learning is a subfield of machine learning that uses multilayer artificial neural networks to learn hierarchical representations from data. It is inspired by the structure of the human brain, where interconnected neurons process information through layers. In deep learning, the term "deep" refers to the presence of multiple hidden layers between the input and output layers. These layers progressively transform raw input data into more abstract and meaningful features.

One of the major advantages of deep learning is automatic feature learning. In traditional machine learning approaches, features often have to be designed manually. For example, in image processing, hand-crafted features such as edges, corners, histograms, and texture descriptors may be extracted and then passed to a separate classifier or decision system. In contrast, deep learning integrates feature extraction and task learning into a single framework. The model learns directly from data which patterns are important for solving the problem.

Deep learning has gained widespread success in:

- image classification
- object detection
- semantic segmentation
- speech recognition
- language translation
- medical image analysis
- image restoration and enhancement

For underwater image enhancement, deep learning is particularly useful because underwater distortions are complex and nonlinear. The relationship between a degraded underwater image and a visually improved version is not simple enough to be captured by a fixed formula in all cases. Different images may require different levels of colour correction, contrast adjustment, or edge recovery. A deep learning model can learn this mapping from examples and adapt more flexibly to the diversity of underwater scenes.

Another advantage of deep learning is that it can model both local and global dependencies. In image enhancement, local features such as edges, textures, and object boundaries must be preserved, while global features such as illumination, contrast, and colour consistency must also be improved. A well-designed deep learning model can balance these objectives through its layered structure.

In this project, deep learning is adopted because:

- underwater degradation is complex and nonlinear
- traditional techniques are often inconsistent across scenes
- learned models can generalize enhancement patterns
- the available datasets are sufficient for supervised learning
- deep networks are highly effective in image-to-image transformation tasks

Thus, deep learning provides the overall methodological foundation of the system, while specific architectures such as CNN define how the data is processed internally.

## 3.3 Why Deep Learning is Suitable for Underwater Image Enhancement

The choice of methodology in any project should be justified based on the problem characteristics. Underwater image enhancement is a challenging task because the degradation process is affected by several physical and environmental factors. Water absorbs light differently depending on wavelength, and scattering by particles reduces scene clarity. As a result, underwater images are affected by multiple distortions at once. A suitable methodology must therefore be capable of learning multidimensional correction patterns.

Deep learning is particularly appropriate for underwater image enhancement for the following reasons.

First, it can learn directly from paired examples. In this project, degraded underwater images and their corresponding target images are available from the EUVP dataset. This makes supervised deep learning a practical choice because the network can compare its predicted output against a known reference and gradually optimize the enhancement mapping.

Second, deep learning handles high-dimensional data effectively. An image contains thousands or millions of pixel values, and their relationships are spatially structured. Deep learning models, especially CNNs, are designed to process this type of structured visual information without flattening away important spatial context.

Third, deep learning models are adaptive. Unlike rule-based enhancement techniques, which may overcorrect some images and undercorrect others, a trained network can learn context-sensitive operations. For example, it may learn stronger colour correction for some blue-dominated images and more moderate correction for others depending on scene composition.

Fourth, deep learning supports end-to-end learning. Instead of performing separate steps for denoising, colour balancing, and sharpening, the network can learn an integrated transformation in a unified model.

Fifth, deep learning methods can be evaluated and improved systematically. Through training logs, validation metrics, qualitative outputs, and ablation studies, the performance of the system can be monitored and refined scientifically.

Because of these advantages, deep learning is not just one option among many; it is a highly suitable methodological framework for the present project.

## 3.4 Overview of Neural Network-Based Methodologies

Before discussing CNN, RNN, and LSTM separately, it is useful to understand that all three belong to the broader family of artificial neural networks. They share some common principles:

- they contain input, hidden, and output layers
- they learn from data by adjusting weights
- they use activation functions to introduce nonlinearity
- they are trained through loss minimization and backpropagation

However, these architectures differ significantly in how they process information.

CNN is designed mainly for spatial data such as images. It uses convolution operations to detect visual patterns and spatial hierarchies. RNN is designed for sequential data and uses recurrent connections to remember previous inputs. LSTM is a more advanced form of RNN that improves the handling of long-term dependencies through memory cells and gating mechanisms.

This difference in structure leads to different application domains. CNN is preferred for tasks such as image classification, enhancement, and segmentation. RNN is preferred for sequence-based tasks such as text modeling and speech recognition. LSTM improves upon RNN for longer and more complex sequences.

In academic methodology writing, it is useful to discuss all three because it demonstrates awareness of the broader deep learning landscape and allows the final algorithm choice to be justified more clearly.

## 3.5 Convolutional Neural Network (CNN)

### 3.5.1 Basic Concept of CNN

Convolutional Neural Network is a deep learning architecture specifically designed to process image data and other grid-like structured inputs. CNN has become one of the most successful and widely used methods in computer vision because it can automatically learn spatial features from raw images. Unlike ordinary fully connected neural networks, CNN does not treat the image as an unstructured vector. Instead, it preserves the two-dimensional spatial arrangement of pixels and learns patterns directly from local neighborhoods.

The fundamental idea behind CNN is that important visual structures such as edges, corners, textures, and shapes can be captured by applying learnable filters across the image. These filters move over the image and compute feature maps that emphasize relevant patterns. Early layers usually detect simple features like edges and colour contrasts, while deeper layers learn more complex structures and scene-level information.

CNNs are highly efficient for image analysis because:

- they exploit spatial locality
- they share weights across the image
- they reduce the number of parameters compared to full connectivity
- they learn hierarchical feature representations

These properties make CNN highly suitable for image enhancement tasks, including underwater image enhancement.

### 3.5.2 Main Components of CNN

A CNN consists of several important components. Understanding these components is necessary for describing how the methodology works.

**1. Convolution Layer**  
The convolution layer is the core building block of CNN. It applies small learnable filters or kernels over the input image. Each kernel detects a specific pattern, such as an edge, colour transition, or texture. The result of convolution is a feature map that highlights where the learned pattern appears in the image.

**2. Activation Function**  
After convolution, an activation function such as ReLU is applied to introduce nonlinearity. Without nonlinearity, the network would behave like a simple linear transformation and would not be able to model complex visual relationships.

**3. Pooling Layer**  
Pooling reduces the spatial size of feature maps while retaining important information. It helps reduce computation and provides some level of translation invariance. Max pooling is commonly used in CNNs. In enhancement tasks, however, pooling should be used carefully because excessive reduction may remove fine image detail.

**4. Batch Normalization**  
Batch normalization helps stabilize and accelerate training by normalizing intermediate activations. It improves convergence and often makes deeper networks easier to train.

**5. Fully Connected or Reconstruction Layers**  
In classification tasks, CNNs often end with fully connected layers. In image enhancement, however, CNNs more commonly use decoder or reconstruction layers to generate an output image rather than a class label.

### 3.5.3 Working Principle of CNN in Images

When an image is given as input to a CNN, the network processes it through a sequence of convolutional transformations. The first layers detect low-level features like edges and colour gradients. Intermediate layers detect textures, shapes, or semantic patterns. Later layers combine these features into more abstract understanding.

In image enhancement tasks, the output is not a class label but another image. Therefore, the network learns how to transform the degraded input image into an improved output image. The network must preserve scene content while improving quality. This means the model should change visual defects without destroying the underlying objects and structures.

For underwater image enhancement, CNN learns corrections such as:

- reducing blue or green colour cast
- improving brightness and contrast
- recovering edges and textures
- enhancing visibility of hidden objects
- reducing haze-like appearance

### 3.5.4 CNN in the Proposed Project

In the present project, CNN is the primary methodology used for enhancement. The system follows an encoder-decoder style CNN architecture with a MobileViT-inspired bottleneck. This means that the input image first passes through encoder layers that extract hierarchical visual features. These features are then processed at a deeper representation stage, and finally decoded back into an enhanced image.

The general architecture used in the project includes:

- convolution-based encoder blocks
- pooling for downsampling
- a bottleneck module for deeper feature reasoning
- decoder blocks for reconstruction
- skip connections to preserve spatial detail
- residual output formulation for stable enhancement learning

This design is highly suitable for underwater enhancement because it allows the model to retain important structural information while learning enhancement corrections.

### 3.5.5 Advantages of CNN

CNN offers several important advantages in this project:

- It is highly effective for image-based tasks.
- It automatically extracts spatial features.
- It preserves image structure better than sequence-based models.
- It supports end-to-end learning from degraded to enhanced image.
- It scales well to large image datasets.
- It works efficiently with GPU acceleration.

Another major benefit is that CNN can capture both local and contextual image patterns. Local distortions such as blur or edge loss can be addressed alongside broader issues like colour imbalance and contrast deficiency.

### 3.5.6 Limitations of CNN

Despite its strengths, CNN also has some limitations:

- It may require large training data to generalize well.
- It can struggle with very long-range dependencies if no contextual module is added.
- It may overfit if regularization or augmentation is insufficient.
- Standard CNNs alone may not fully capture global scene context.

These limitations explain why modern image enhancement systems often extend CNNs with attention, transformer, or hybrid modules. In this project, the MobileViT-style bottleneck helps address some contextual limitations while preserving the CNN foundation.

### 3.5.7 Why CNN is Most Suitable Here

CNN is the most suitable methodology for this project because underwater image enhancement is fundamentally an image-to-image transformation problem. The system must process spatial information, preserve object boundaries, and improve visual quality at pixel and feature levels. CNN is naturally designed for this purpose. It directly models spatial correlations in images and is therefore more appropriate than sequence-oriented architectures such as RNN and LSTM.

## 3.6 Recurrent Neural Network (RNN)

### 3.6.1 Basic Concept of RNN

Recurrent Neural Network is a deep learning architecture primarily designed for sequential data. Unlike feedforward neural networks, RNN contains recurrent connections that allow information from previous time steps to influence the current computation. This gives the network a form of memory and makes it useful for tasks where order and temporal dependence matter.

RNN is widely used in:

- natural language processing
- text generation
- speech recognition
- machine translation
- time-series prediction
- sequential sensor analysis

In RNN, the hidden state at each step is updated using both the current input and the previous hidden state. This allows the network to maintain contextual information across a sequence.

### 3.6.2 Working Principle of RNN

The working principle of RNN can be understood as repeated processing over a sequence. At each time step:

1. the network receives the current input
2. it combines this input with the previous hidden state
3. it computes a new hidden state
4. it produces an output if required

Because the same weights are reused at each time step, the network can process sequences of varying lengths. This weight sharing is efficient and allows temporal pattern learning.

### 3.6.3 Strengths of RNN

RNN is powerful when the current output depends on what came before. For example:

- in language, the meaning of a word depends on previous words
- in speech, the current sound depends on neighboring sounds
- in time-series data, the present value depends on past values

The main strengths of RNN include:

- ability to handle sequence data
- memory of previous inputs
- support for variable-length sequences
- useful modeling of temporal dependencies

### 3.6.4 Limitations of RNN

Although RNN is theoretically powerful, in practice it has serious limitations. The most important issue is difficulty in learning long-term dependencies. During backpropagation through many time steps, gradients can become very small or very large. This is known as the vanishing gradient or exploding gradient problem. As a result, standard RNN often struggles to remember information from much earlier in the sequence.

Other limitations include:

- unstable learning for long sequences
- difficulty in parallelization
- weaker handling of very complex contextual memory

These limitations motivated the development of LSTM and GRU architectures.

### 3.6.5 Relevance of RNN to the Present Project

RNN is not the primary methodology for the proposed underwater image enhancement system because the project deals with static image enhancement rather than sequence prediction. A still underwater image does not naturally form a temporal sequence in the way that words or time steps do. Therefore, the recurrent memory mechanism of RNN is not directly needed for the current system.

However, discussing RNN is still useful in the methodology chapter for several reasons:

- it shows understanding of broader deep learning architectures
- it allows comparison between spatial and sequential models
- it helps justify why CNN is chosen instead

If the project were extended to underwater video enhancement, frame sequence restoration, or underwater navigation sensor fusion, RNN could become more relevant. In such cases, the temporal relation between consecutive frames or measurements could be learned using recurrent methodology.

Thus, RNN is academically important but practically secondary for this project.

## 3.7 Long Short-Term Memory (LSTM)

### 3.7.1 Basic Concept of LSTM

Long Short-Term Memory is a specialized form of recurrent neural network introduced to overcome the limitations of standard RNN. LSTM was designed to preserve relevant information over longer sequences and prevent the rapid loss of gradient information during training.

The key innovation in LSTM is the memory cell and gating mechanism. Instead of updating hidden state in a simple recurrent manner, LSTM carefully controls what information should be stored, forgotten, or passed forward. This makes it much more effective than simple RNN in capturing long-range dependencies.

### 3.7.2 Main Components of LSTM

LSTM typically contains three gates:

**1. Input Gate**  
This gate determines how much of the new input should be written into memory.

**2. Forget Gate**  
This gate decides what old information should be discarded from memory.

**3. Output Gate**  
This gate controls how much of the memory should be exposed as the current output or hidden state.

These gates allow the model to regulate information flow intelligently. As a result, LSTM can remember useful context over longer intervals than a standard RNN.

### 3.7.3 Strengths of LSTM

LSTM has several important strengths:

- it handles long-term dependencies better than standard RNN
- it reduces vanishing gradient problems
- it is effective for long and complex sequences
- it improves temporal modeling stability

For these reasons, LSTM has been widely used in:

- language modeling
- speech recognition
- time-series forecasting
- handwriting recognition
- anomaly detection in sequential data

### 3.7.4 Limitations of LSTM

Although LSTM is more powerful than standard RNN, it also has limitations:

- it is computationally more expensive than simple RNN
- it is more complex to train and tune
- it is less naturally suited to spatial image modeling than CNN
- it may be unnecessary for tasks without meaningful sequential dependency

Thus, while LSTM is a strong sequence model, it is not automatically the best choice for every deep learning problem.

### 3.7.5 Relevance of LSTM to the Present Project

In this underwater image enhancement project, LSTM is not directly used as the main algorithm because the enhancement task is performed on static images rather than on temporal sequences. The main challenge in the project is correcting spatial visual degradation, not remembering long sequential patterns. Therefore, LSTM is less suitable than CNN for the current objective.

Still, LSTM remains relevant as a theoretical methodology because it represents an important evolution of recurrent learning. If the project were expanded into underwater video processing, sequential scene interpretation, or temporal enhancement across video frames, LSTM could help maintain consistency over time and reduce flickering between frames.

This comparative discussion helps demonstrate why the choice of CNN is methodologically justified.

## 3.8 Comparative Analysis of CNN, RNN, and LSTM

In order to justify the final methodological choice more clearly, it is useful to compare CNN, RNN, and LSTM across several dimensions.

### 3.8.1 Nature of Input Data

- CNN is best suited for spatial data such as images.
- RNN is best suited for sequential data such as text or time steps.
- LSTM is also suited for sequential data, especially when long-term memory is needed.

Since the project deals with static underwater images, CNN naturally aligns with the input type.

### 3.8.2 Feature Learning Style

- CNN learns local and hierarchical spatial features.
- RNN learns temporal dependencies between ordered inputs.
- LSTM learns long-range temporal dependencies using gated memory.

The enhancement problem depends mainly on spatial correction, so CNN has clear methodological advantage.

### 3.8.3 Typical Applications

- CNN: image classification, segmentation, restoration, enhancement
- RNN: text generation, sequence labeling, speech modeling
- LSTM: translation, long text modeling, time-series forecasting

Among these application categories, underwater image enhancement clearly falls under the CNN domain.

### 3.8.4 Strength in This Project Context

- CNN preserves image structure and supports image-to-image mapping.
- RNN provides unnecessary temporal memory for a static image task.
- LSTM offers advanced sequence modeling but does not directly address spatial enhancement needs.

### 3.8.5 Computational Consideration

CNN is generally more straightforward and efficient for image tasks, especially with GPU support. RNN and LSTM often involve sequential processing that is harder to parallelize efficiently. For a practical academic project with image datasets and enhancement outputs, CNN is both conceptually and computationally appropriate.

This comparative analysis strongly supports the conclusion that CNN is the most suitable methodology for the proposed system.

## 3.9 Methodological Flow of the Proposed Project

The practical methodology of the project can be described in a sequence of stages that combine theory and implementation.

### 3.9.1 Data Acquisition

The first stage involves obtaining underwater datasets such as EUVP, UIEB, and RUIE. These datasets provide degraded underwater images and, in some cases, corresponding reference images. The paired EUVP dataset is the main training source used to teach the model the enhancement mapping.

### 3.9.2 Data Preprocessing

The collected images are then preprocessed by:

- checking directory validity
- filtering image formats
- matching input and target filenames
- converting images to RGB
- resizing them to 256 x 256
- applying augmentation where required
- converting them into tensors for model input

This stage ensures that the dataset is standardized and suitable for deep learning.

### 3.9.3 Model Learning

The core CNN-based enhancement model receives degraded underwater images as input. During training, the network produces an enhanced output and compares it with the target image. The difference is measured through loss functions, and model parameters are updated using backpropagation and optimization.

### 3.9.4 Validation and Evaluation

The system validates its performance on reserved validation images and later evaluates on EUVP, UIEB, and RUIE using underwater image quality metrics such as UCIQE and UIQM, as well as full-reference metrics like PSNR and SSIM where possible.

### 3.9.5 Qualitative Interpretation

Finally, the enhanced results are visually examined using comparison panels and publication-style figures. This stage helps verify whether the outputs are visually meaningful in addition to being numerically improved.

This methodological flow shows that the project is not merely based on one algorithmic idea. Rather, it is a complete experimental methodology integrating data handling, model training, evaluation, and interpretation.

## 3.10 Role of Loss Functions and Optimization in the Methodology

A complete methodology chapter should also briefly mention how learning is guided. In deep learning, the algorithm does not improve automatically; it needs a loss function and an optimizer.

In the present project, the model is trained using a combined loss strategy that includes:

- L1 loss
- edge loss
- colour loss

This choice is methodologically important because underwater image enhancement requires more than simple pixel matching. L1 loss encourages overall output similarity to the target. Edge loss promotes structural sharpness and preserves boundaries. Colour loss helps reduce unwanted global colour cast and encourages better channel balance.

The optimizer updates the network parameters based on the loss. By iteratively minimizing this combined loss, the CNN-based model learns how to enhance degraded underwater inputs in a visually meaningful way.

This point can be included in the report to demonstrate that methodology includes not only the choice of network architecture, but also the learning objective used to train it.

## 3.11 Why the Proposed Methodology is Appropriate

The methodology selected for the project is appropriate because it aligns closely with the problem requirements. Underwater image enhancement requires:

- processing of spatial image information
- preservation of local textures and edges
- correction of colour and contrast distortions
- ability to learn from paired examples
- support for end-to-end enhancement

CNN-based methodology satisfies all of these conditions. It is specifically designed for images, supports feature hierarchy learning, and integrates naturally with encoder-decoder enhancement frameworks.

RNN and LSTM, although powerful in other settings, are less aligned with the current problem because they are specialized for sequential relationships rather than spatial restoration. Including them in the methodology chapter still has academic value because it demonstrates analytical comparison and supports the justification of the final algorithmic choice.

The chosen methodology is also appropriate from a practical standpoint:

- the datasets are image-based and compatible with CNN
- the hardware and framework support efficient CNN training
- evaluation metrics are suited to image enhancement output
- the model architecture can be extended or refined in future work

Thus, the project methodology is both theoretically sound and practically implementable.

## 3.12 Limitations of the Methodological Scope

No methodology is perfect, and acknowledging limitations improves the quality of the report. The methodology used in this project has some constraints.

First, the analysis focuses mainly on three major deep learning architectures and not on every possible modern alternative. Other approaches such as GANs, attention-based transformers, diffusion models, and physics-based underwater restoration methods are not discussed in detail here.

Second, the project emphasizes static image enhancement. Therefore, sequence-based models such as RNN and LSTM are discussed mostly from a conceptual perspective rather than through implementation.

Third, the performance of the CNN methodology depends strongly on dataset quality and diversity. If the training data is not representative enough, the model may not generalize to all underwater conditions.

Fourth, deep learning methodology often requires significant computational resources compared to classical image processing methods. Although the proposed system is relatively lightweight, training still depends on a suitable hardware environment.

These limitations do not weaken the project, but they provide a realistic view of the methodological boundaries.

## 3.13 Future Methodological Extensions

The methodology used in this project can be expanded in future work. Some possible extensions include:

- applying CNN-LSTM hybrid methods for underwater video enhancement
- using transformer-based global attention modules
- integrating generative adversarial learning for more realistic enhancement
- incorporating perceptual loss functions
- introducing multi-scale feature fusion for better detail recovery
- extending the system to real-time underwater enhancement in robotics

These directions show that the current methodology forms a solid foundation upon which more advanced work can be built.

## 3.14 Conclusion

The methodology chapter provides the theoretical and practical foundation of the proposed underwater image enhancement system. Deep learning has been identified as the primary methodological framework because of its ability to learn complex feature representations and nonlinear mappings from degraded images to enhanced outputs. Among the deep learning algorithms discussed, Convolutional Neural Network, Recurrent Neural Network, and Long Short-Term Memory each have important roles in modern artificial intelligence, but they differ significantly in their areas of application.

CNN is the most suitable algorithm for the present project because underwater image enhancement is fundamentally a spatial image-processing task. CNN is designed to learn features directly from images, preserve spatial structure, and support end-to-end image transformation. It can automatically detect edges, textures, colour patterns, and visual distortions, making it highly effective for enhancement tasks. In the proposed system, CNN serves as the core methodological engine of the image enhancement model.

RNN is an important methodology for sequence learning and temporal modeling, but it is not the most appropriate choice for static underwater image enhancement. LSTM improves upon RNN by introducing memory cells and gating mechanisms, making it powerful for long sequences, but it still remains more suitable for time-dependent tasks than for still-image restoration. Their inclusion in this chapter is academically useful because it supports comparative analysis and clarifies why CNN is selected over alternative neural network families.

Overall, the methodology adopted in this project is logically justified, technically relevant, and aligned with the nature of the problem. The chapter establishes a strong theoretical base for the implementation, training, evaluation, and analysis presented in the later parts of the report. It also demonstrates that the proposed underwater image enhancement system is built upon a sound understanding of deep learning methodologies and their application domains.
