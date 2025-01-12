# CE320

## Modules to be installed
```bash
pip install tokenize
pip install matplotlib
pip install scikit-learn
pip install seaborn
pip install scipy
```

## Phase One:

Step 1:
### tokenization.py
//Tokenization: Use Python's tokenize module to parse Python code.
//This module will break down the code into tokens, such as keywords, identifiers, operators, and literals, 
// effectively preparing the code for N-gram analysis.

Step 2:
### n_gram.py
//N-gram Generation: Implement functions to generate N-grams from the tokenized code. 
//These functions should handle different values of N (1-grams, 2-grams initially). 
// The output should be sequences of N consecutive tokens.


Step 3:
### n_gram_enchanced.py
// Flexibility: Design the N-gram generation function to be easily adaptable for different N values. 
// This way you can extend your application to higher N-grams later.


Step 4: 
### frequency_counting.py
// Frequency Counting: Create a mechanism to count the frequency of each generated N-gram in the code.
// You can use dictionaries or similar data structures to store the counts..


Step 5: 
### optimized_ngram_frequency_counting.py
// Efficiency: Optimize the counting process to be efficient, 
// as large codebases might produce a significant number of N-grams.

Step 6:
### frequency_matrices.py
// Matrix Creation: Generate frequency matrices for 1-gram and 2-gram analysis. 
// These matrices will represent the frequency of each N-gram in the analyzed code of different programmers.

Step 7:
### matrices_representation.py
// Data Structure: Carefully choose the appropriate data structure 
// to represent these matrices for efficient storage and comparison.

Step 8:
### degree_of_similarity.py
// Comparison: Implement a function to compare the N-gram frequency matrices of different programmers14. 
// This will establish the degree of similarity in their programming styles4.

Step 9:
### similarity_metric.py
Similarity Metric: Consider using appropriate similarity metrics
(e.g., cosine similarity, Euclidean distance) for a numerical comparison of programming styles.

Step 10:
### code_analysis.py
Input files folders containing different programmer's code.
read the codes from the files and analyse them.
Output path for storing the analysis results.



## Phase Two:

Data Analysis and Homogeneity:

Pairwise Comparison: 

The program will need to compare all pairs of programmers in a team to determine the similarity of their programming styles.

Homogeneity Determination: Develop a way to determine the degree of homogeneity within a team based on the pairwise comparisons.

For example, you can calculate the average or variance of similarity scores to assess team homogeneity.

Output: Provide clear and concise output on the similarity scores of each pair of programmers, as well as the overall team homogeneity score.
