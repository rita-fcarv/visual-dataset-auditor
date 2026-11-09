# GPU-Accelerated Visual Dataset Auditor

**A CUDA/C++ application for evaluating image dataset quality and detecting visually similar images using GPU-accelerated processing.**

## Overview

Image datasets are fundamental to computer vision and machine learning applications. However, building and maintaining high-quality datasets can be challenging, particularly when working with large collections of images.

Datasets may contain duplicate or near-duplicate images, blurry samples, or images with poor brightness and contrast. These issues can affect downstream analysis, introduce redundancy into training datasets, and make data preparation more time-consuming.

The **GPU-Accelerated Visual Dataset Auditor** was developed to help address these challenges by providing an automated way to analyse image collections and generate reports highlighting potential data quality issues.

The project combines traditional image processing techniques with custom CUDA kernels to explore how GPU parallelism can improve the efficiency of dataset auditing.

Rather than automatically deciding which images should be removed, the application provides objective measurements and similarity information to support informed dataset curation.

## Project Goals

The project has three main goals:

1. **Assess image quality:** Extract meaningful metrics that help identify potentially blurry, poorly exposed, or low-contrast images.
2. **Detect duplicate and similar images:** Use perceptual hashing to identify image pairs that may contain redundant visual information.
3. **Evaluate GPU acceleration:** Implement image processing and comparison algorithms using CUDA, validate their correctness, and compare their performance against CPU implementations.

Together, these objectives demonstrate how GPU programming can support practical computer vision data preparation workflows.

## How It Works

The application processes a directory of images through a sequence of GPU-accelerated operations.

### 1. Image Loading and Preprocessing

Images are loaded using OpenCV and converted from BGR to grayscale using a custom CUDA kernel.

Grayscale conversion simplifies subsequent analysis by representing each pixel with a single intensity value.

### 2. Image Quality Assessment

For each image, the application computes three quality metrics:

| Metric | Description | Purpose |
|---|---|---|
| Mean brightness | Average grayscale pixel intensity | Identify potentially underexposed or overexposed images |
| Contrast | Standard deviation of pixel intensities | Identify images with limited intensity variation |
| Blur score | Variance of the Laplacian response | Estimate sharpness and identify potentially blurry images |

These metrics are calculated using custom CUDA kernels and parallel reductions.

The resulting values are saved to a CSV report for further inspection.

Importantly, these measurements are indicators rather than definitive quality classifications. For example, a naturally dark image may have low brightness without being incorrectly exposed.

### 3. Perceptual Image Hashing

To detect visually similar images, the application computes a 64-bit difference hash (*dHash*) for each image.

The process involves resizing the grayscale image to 9×8 pixels, comparing adjacent pixel intensities, and encoding the comparisons into a compact binary hash.

Unlike cryptographic hashing, perceptual hashing is designed to capture aspects of visual structure. Images with similar visual content may therefore produce similar hashes even when their pixel values are not identical.

### 4. GPU Similarity Detection

After generating the hashes, the application compares image pairs using their Hamming distance, which counts the number of differing bits between two hashes.

Pairs with distances below or equal to a configurable threshold are reported as potentially similar.

The comparison is accelerated using CUDA, allowing multiple image pairs to be evaluated in parallel.

The optimised implementation processes unique image pairs and collects matching results on the GPU, avoiding the need to transfer a complete pairwise distance matrix back to the CPU.

### 5. Report Generation

The application generates two main reports:

- **Quality report:** Contains the image quality metrics for each successfully processed image.
- **Similarity report:** Contains pairs of potentially similar images and their Hamming distances.

These reports can be used to inspect image collections, investigate potential duplicates, and support decisions about dataset cleaning.

## Technologies

| Technology | Role |
|---|---|
| CUDA | GPU kernels, parallel processing and reductions |
| C++14 | Main application and processing logic |
| OpenCV | Image loading, resizing and output |
| NVIDIA GPU | Hardware acceleration |
| Python 3 | Synthetic dataset generation and evaluation |
| GNU Make | Building the application |

The project was developed and tested in the following environment:

- GPU: NVIDIA L4
- CUDA Toolkit: 11.3
- GCC: 9.3
- OpenCV: 4.2

The supplied Makefile uses the `sm_86` CUDA architecture target, which is compatible with the tested environment. Other GPU/toolkit combinations may require adjusting this flag.

## Installation and Build

### Prerequisites

A Linux environment with the following dependencies is required:

- An NVIDIA GPU with compatible drivers
- CUDA Toolkit and `nvcc`
- A C++14-compatible compiler
- OpenCV 4 development libraries
- GNU Make
- Python 3

### Build the application

From the project root:

```bash
make clean
make
```

This generates the `visual_auditor` executable.

## Usage

### Run the auditor

```bash
./visual_auditor \
  --input data/images \
  --output results/processed \
  --threshold 5
```

The application accepts the following arguments:

| Argument | Description |
|---|---|
| `--input` | Directory containing the input images |
| `--output` | Directory where reports and processed outputs are saved |
| `--threshold` | Maximum Hamming distance used to identify similar images (0–64) |

The threshold is configurable because the appropriate level of similarity depends on the characteristics of the dataset.

A threshold of 5 was selected for the project's synthetic robustness evaluation. It should not be treated as universally optimal.

### Generate the test datasets

The repository includes Python scripts for generating synthetic datasets:

```bash
python3 scripts/generate_dataset.py
python3 scripts/generate_validation.py
python3 scripts/generate_robustness_dataset.py
```

These scripts create reproducible image collections for testing the application, including controlled variations such as brightness changes, blur and resizing.

### Run the evaluation

```bash
./visual_auditor \
  --input data/robustness \
  --output results/optimized_validation \
  --threshold 5

python3 scripts/evaluate_similarity.py \
  --input data/robustness \
  --report results/optimized_validation/similarity_report.csv
```

The evaluation compares detected image pairs against the expected relationships in the synthetic dataset.

### Run the automated workflow

A convenience script is also provided:

```bash
./run.sh
```

It builds the application, generates the test datasets, runs the auditor and evaluates similarity detection.

## Validation and Experimental Results

Correctness and performance were evaluated separately to ensure that GPU acceleration did not compromise the expected results.

### Similarity Detection Accuracy

A synthetic robustness dataset was created with 120 images organised into groups containing controlled image variations.

The evaluation includes exact copies, brightness adjustments, blur, resizing and visually different images.

At a Hamming distance threshold of 5, the results were:

| Metric | Result |
|---|---:|
| Images | 120 |
| Possible image pairs | 7,140 |
| Expected similar pairs | 200 |
| Detected similar pairs | 200 |
| True positives | 200 |
| False positives | 0 |
| False negatives | 0 |
| True negatives | 6,940 |
| Precision | 1.0000 |
| Recall | 1.0000 |
| F1 score | 1.0000 |

These results demonstrate correct detection against the defined synthetic ground truth. They do not imply perfect accuracy on arbitrary real-world image datasets.

### CPU vs GPU: Image Processing

Grayscale conversion and image quality metrics were benchmarked at three image resolutions.

| Resolution | Grayscale speedup | Quality metrics speedup |
|---|---:|---:|
| 128×128 | 0.15× | 0.21× |
| 512×512 | 1.52× | 1.18× |
| 1024×1024 | **2.36×** | **2.79×** |

For 1024×1024 images:

| Operation | CPU | GPU |
|---|---:|---:|
| Grayscale conversion | 2.2104 ms | 0.9384 ms |
| Quality metrics | 3.2144 ms | 1.1507 ms |

The GPU provides measurable benefits for larger images, while the CPU remains faster for smaller images because GPU allocation, kernel launches and data transfers introduce additional overhead.

Grayscale outputs were also compared for correctness. At 1024×1024 resolution, only 64 pixels differed between the CPU and GPU implementations, with a maximum difference of one intensity level.

### CPU vs GPU: Similarity Detection

The similarity detection algorithm was benchmarked using synthetic 64-bit image hashes.

| Images | CPU | Original GPU | Optimised GPU | Optimised speedup vs CPU |
|---|---:|---:|---:|---:|
| 120 | 0.0233 ms | 0.2418 ms | 0.2420 ms | 0.10× |
| 500 | 0.3883 ms | 0.5496 ms | 0.2443 ms | 1.59× |
| 1,000 | 1.5476 ms | 1.5796 ms | 0.2625 ms | **5.89×** |

The original GPU implementation transferred the full pairwise distance matrix back to the CPU.

The optimised implementation reduces unnecessary data transfers by collecting only matching pairs, resulting in a substantial improvement for larger datasets.

At 1,000 images, the optimised GPU implementation achieved approximately a **5.89× speedup over the CPU baseline** and reduced measured execution time by approximately 83.4% compared with the original GPU implementation.

All benchmarks were performed in the NVIDIA L4 environment using synthetic workloads. The GPU timings include memory allocation and transfer overhead, and the CPU reference implementations are single-threaded. Results may differ with other hardware, datasets and implementations.

## Implementation Challenges and Lessons Learned

### GPU Overhead and Workload Size

One important observation was that GPU acceleration is not automatically beneficial for every workload.

For small images and datasets, the cost of allocating GPU memory, transferring data and launching kernels can exceed the benefits of parallel execution.

As the workload increases, GPU parallelism becomes more effective.

### Memory Transfer Optimisation

The initial similarity implementation transferred a complete matrix of pairwise distances to the CPU.

Although the comparisons were performed in parallel, transferring unnecessary results introduced substantial overhead.

The optimised implementation instead evaluates unique image pairs and compacts matching results on the GPU.

This change demonstrated how memory movement and data representation can be as important as kernel execution speed.

### CPU and GPU Numerical Consistency

CPU and GPU implementations can produce slightly different floating-point results because of differences in arithmetic evaluation and rounding.

During grayscale validation, a small number of pixels differed by one intensity level.

This highlighted the importance of using appropriate numerical tolerances when validating equivalent implementations rather than requiring exact bit-for-bit equality in every case.

### Correctness Before Performance

Performance improvements were evaluated only after checking that the implementation produced the expected results.

The project uses controlled synthetic datasets and explicit similarity ground truth to distinguish correctness from execution speed.

This approach helped ensure that optimisation did not silently change the detector's behaviour.

## Limitations and Future Improvements

Although the application demonstrates useful functionality and measurable acceleration, several limitations remain.

**Perceptual hashing:** dHash is a lightweight similarity technique, but it is not robust to every visual transformation. Certain changes may lead to missed matches, while unrelated images may occasionally produce similar hashes.

**Scalability:** Pairwise comparison has quadratic computational complexity. Very large datasets would benefit from candidate filtering, indexing or approximate nearest-neighbour techniques.

**GPU memory management:** Reusing device buffers and processing images in batches could further reduce allocation and transfer overhead.

**Quality assessment:** Brightness, contrast and Laplacian variance provide useful indicators but do not capture all aspects of image quality.

**Dataset diversity:** The current validation relies primarily on synthetic datasets. Evaluation using real-world image collections would provide a more representative assessment of practical performance.

Potential future extensions include more advanced perceptual similarity methods, configurable quality filters, interactive report visualisation and support for larger datasets through streaming or batched processing.

## Project Structure

```text
visual-dataset-auditor/
├── src/
│   ├── main.cu
│   ├── image_processor.cu
│   ├── image_processor.cuh
│   ├── image_metrics.cu
│   ├── image_metrics.cuh
│   ├── image_hash.cu
│   ├── image_hash.cuh
│   ├── similarity.cu
│   └── similarity.cuh
├── benchmarks/
│   ├── benchmark_image.cu
│   └── benchmark_similarity.cu
├── scripts/
│   ├── generate_dataset.py
│   ├── generate_validation.py
│   ├── generate_robustness_dataset.py
│   └── evaluate_similarity.py
├── data/
├── results/
│   └── benchmarks/
├── submission/
├── Makefile
├── run.sh
└── README.md
```

Generated datasets, reports and benchmark outputs are stored in their respective directories. The `submission/` directory contains supporting material for the Coursera capstone submission.

## Conclusion

The GPU-Accelerated Visual Dataset Auditor demonstrates how CUDA can be applied to a practical computer vision data preparation problem.

By combining image quality assessment, perceptual hashing and parallel similarity detection, the project provides a foundation for inspecting image collections and identifying potential issues before they are used in downstream workflows.

The experimental results show that GPU acceleration can offer meaningful performance improvements, particularly for larger images and datasets. At the same time, the project illustrates that efficient GPU programming requires careful consideration of workload size, memory transfers, numerical correctness and algorithm design.

Overall, the project connects GPU programming concepts with a real-world dataset quality challenge while providing a reproducible framework for further experimentation and optimisation.