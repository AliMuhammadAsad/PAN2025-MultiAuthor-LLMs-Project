# PAN2025-MultiAuthor-LLMs-Project

This repo contains the project for the course ```CS/CE 335/466 - Introduction to Large Language Models```, for which we are working on CLEF 2025 - PAN Track Multi-Autho Analysis - Style Change Detection task which can be found [https://pan.webis.de/clef25/pan25-web/style-change-detection.html#related-work](here).

---

### Project Status

- **Literature Review**: Completed
- **Data Collection**: Completed
- **Data Preprocessing**: Completed
- **Model Training**: _In Progress_
- **Model Evaluation**: _In Progress_
- **Model Deployment**: _Pending_
- **Final Report**: _Pending_
- **Final PAN Submission**: _Pending_

---

### Output Structure:
For each of the implementation of the models, the output structure should be as follows:
```
output_model/easy/solution-problem-X.json
output_model/medium/solution-problem-X.json
output_model/hard/solution-problem-X.json
```
where X is the problem number.

---

### Verifying the Output:
To verify the output, change the directory to the ```verifier``` folder and run the following command:
```
python verifier.py --output <path_to_output_file> --input dataset/
```

If there are no issues with the output, the verifier should print it read 900 problem ids from each level of the dataset. If there are any issues, the verifier will print the problem ids which are not present in the output, or there was an Error. 

---

### Evaluating the Output:
To evaluate the output, change the directory to the ```evaluator``` folder and run the following command:
```
python evaluator.py --predictions <path_to_output_file> --truth dataset/ --output <path_to_output_file>
```
where ```<path_to_output_file>``` is the path to the output file where the evaluation results will be stored. The results are stored as a .prototext file.

---

### Contributors:

- Ali Muhammad Asad 
- Musaib
- Syed Muhammad Areeb Kazmi
- Sarim Tahir

---

### Acknowledgements:

Special thanks to our course instructors, Dr. Abdul Samad and Dr. Faisal Alvi, and Sandesh Kumar (our course Research Assistant) for their guidance and support throughout the project.
