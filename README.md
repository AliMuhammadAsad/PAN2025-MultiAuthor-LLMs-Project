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

### Dataset

The dataset is divided into three levels: easy, medium and hard, where each level is split into three parts:
- _training set_: Contains 70% of the whole dataset and includes ground truth data. This data would be used to develop and train the models.
- _validation set_: Contains 15% of the whole dataset and includes ground truth data. This data would be used to evaluate and optimize the models.
- _test set_: Contains 15% of the whole dataset, no ground truth data is given. This set is held by the organizers and would be used to evaluate the models.

**Input Format:**
For each problem instance X (i.e., each input document), two files are provided:
1. *problem-X.txt*: contains the actual text.
2. *truth-problem-X.json*: contains the ground truth, i.e., the correct solution in JSON format.

Examples for the above formats are as so:

sample.txt file
```
Do you understand what you are saying here?
This would be reasonable if WW3 wasn't one of the very possible and very few outcomes and if Ukraine winning the war with Russia's unconditional surrender was actually a possible outcome.
What you are saying is you think WW3 or Ukrainians being genocided completely is better than making concessions.
Concessions and peace negotiations should absolutely, positively be on the table.
You do not want war.
Ask someone who has seen war if war is a noble or fun thing to be a part of.
Not to continue ranting on this point, but do you want a frag grenade dropped on you by a drone while you are shitting in a frozen ditch?
That's what is happening right now to soldiers on both sides in Ukraine.
You especially do not war with a nuclear power headed by a psychopath who doesn't give a fuck and wishes to re-implement a Russian empire by whatever means possible.
Exactly.
Right now Ukraine is holding its own against Russia.
That's a small miracle even with Western equipment.
If they give Putin a chance to recuperate he'll mass forces and steamroll them no matter the cost in Russian soldiers lives.
```

sample.json file:
```
{
    "authors": 2,
    "changes": [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
}
```

The result (key "changes") is represented as an array, holding a binary for each pair of consecutive sentences within the document (0 if there was no style change, 1 if there was a style change).

---

### Results (so far):

| Model | F1 Score (Easy) | F1 Score (Medium) | F1 Score (Hard) |
|-------|-----------------|-------------------|-----------------|
| Bag of Words (Baseline) | 0.6531 | 0.6498 | 0.5725 |
| Bag of Words (Improved) | 0.7910 | 0.6657 | 0.6125 |
| BERT | 0.9412 | 0.7879 | 0.7547 |

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
