# Domain-Enhanced-Dual-Branch-Model-for-Efficient-and-Interpretable-Accident-Anticipation

1. Environment setup

Install the conda environment. Please note that not all configurations in the provided environment file are required.

2. Dataset download:

Download the video dataset from the DAD official page and split it according to the official introduction (https://github.com/MoonBlvd/tad-IROS2019).

3. Download accident report:

Download all accident report PDF files from the DMV official website "https://www.dmv.ca.gov/portal/vehicle-industry-services/autonomous-vehicles/autonomous-vehicle-collision-reports/"

Extract the content of "key == 'ADDRESS_2.1.0.1'" from the PDF file and store it in "accident.json"

4. Non-accident report generation:

Use GPT-4o to upload "accident.json", and use the prompts provided by the 'prompts' file to generate a non-accident report, and save it in "non_accident.json".

5. Text feature extraction:

Deploy Long-CLIP (https://github.com/beichenzbc/Long-CLIP), use ‘longclip.tokenize’ and ‘model.encode_text’ to extract all text features in “accident.json” and “non_accident.json”, and use max pooling to compress to [2, 512].

6. Video feature extraction:

Use the extractor script provided in the code library to extract video features.

7. Modify all input and output paths to your own path.

8. Run the experiment.
