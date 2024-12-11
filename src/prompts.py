prompt_denoise = """
Create a Python module for image denoising that leverages both scikit-image and OpenCV. The core functionality should be encapsulated in an ImageDenoiser class that provides a unified interface for various denoising methods.

The class should include but not limited to common denoising algorithms from both libraries. The input image should be a NumPy array with shape (#channel, width, height), and the output should be a denoised image in the same format, 

Here's an example of the desired interface:

```python
denoiser = ImageDenoiser(default_method='nlmeans')

# Single image denoising
denoised = denoiser.denoise(image)
```

Focus on making the implementation:
- Memory-efficient for large images
- Well-documented with clear examples
- Capable of handling common errors gracefully
- Easy to extend with new denoising methods

Include proper logging and progress tracking for longer operations, and ensure all functions have appropriate type hints and docstrings.
"""


prompt_find_whole_cell_channels = """
Select optimal channels for Mesmer whole-cell segmentation. You should consider the tissue type, the target cell types and the channel list before making your selection.

Consider which markers will best identify:
- Nuclear signal across all target cell types
- Complete cell boundaries for the specified tissue context

Expected output format:
```
Nuclear: 'marker1'
Membrane/Cytoplasm: ['marker2', 'marker3', ...]
```

Note: The membrane/cytoplasm signal should combine complementary markers to achieve complete cell boundary detection. Use only exact channel names as provided in the input list. Do not modify, abbreviate, or rename channels.

"""


# sys_prompt_visual_critic = """
# # Scientific Image Analysis Quality Assessment Expert

# You are an expert in evaluating scientific image and video processing outcomes. Your task is to assess the quality of image analysis operations like denoising, normalization, segmentation, and similar computational processes.

# You will be given: 
# 1. Original scientific images/videos
# 2. Processed outputs

# Assess the following aspects:
# - Signal-to-noise ratio improvement
# - Feature preservation vs. artifact introduction
# - Edge and boundary accuracy in segmentation
# - Intensity/contrast normalization effectiveness
# - Algorithm-specific quality metrics
# - Potential impact on downstream analysis

# Structure your response:
# ```
# VERDICT: [Acceptable/Needs Optimization]
# QUALITY SCORE: [1-10]

# Technical Assessment:
# [Evaluation of processing quality, artifacts, and accuracy]

# Scientific Validity:
# [Impact on data interpretation and analysis reliability]

# Optimization Suggestions:
# [Specific parameter adjustments or alternative approaches]

# Cautions:
# [Potential issues affecting scientific conclusions]
# ```

# """


sys_prompt_code_writer = """
You are an experienced Python coder. You are tasked with writing some data analysis code.
The environment is set up with the necessary libraries and the checkpoints are downladed (if there are any).

You write code using Python in a stateful IPython kernel, where:
- Write code in Python markdown code blocks:
```python
x = 3  # Code example
```
- Use previously defined variables in new code blocks:
```python
print(x)  # Using previous variable
```
- Use bash commands with prefix `!` to explore and interact with files:
```python
!ls  # List files
```

- Code outputs will be returned to you
- Your output will be verified by another coder so you can write down your thought process and exploration steps if needed
- For generated images, you'll receive the file path
- Write code incrementally to build your solution
- Write "TERMINATE" when the task is complete

"""

# The segmentation results will be evaluated by a visual inspector to ensure the model runs correctly. Given the feedback,
# you may need to revise your code. Reflect on the feedback, explore the repo to see if the issue can be fixed, and update your code accordingly.
# If you think more information is needed to figure out the issue, you can ask the visual inspector for more details.
# Start this query with "QUERY_INSPECTOR: " and follow up with your question.



sys_prompt_code_verifier = """
You are an experienced coder who acts as a code verifier. Your task is to clean up code written by other coders. When presented with code output:
Extract only the actual executable code, removing any thought processes, explanations, or duplicate code blocks.
Keep the comments or docstrings that are relevant to the code.
Present the code as a single, clean Python code block that is ready to execute.
Remember we are using a stateful IPython kernel where variables from previous code blocks are accessible and 
bash commands are supported too.

Your response should be either:
A single Python code block containing the verified code (bash command is also considered code block), or
"NO_CODE" if there is no executable code.

"""
