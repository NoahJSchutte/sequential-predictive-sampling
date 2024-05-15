# sequential-predictive-sampling
Codebase for: Improving Single Solution Metaheuristic Efficiency for Stochastic Optimization by Sequential Predictive Sampling - Noah Schutte, Krzysztof Postek and Neil Yorke-Smith

Article: (to be published at CPAIOR 2024)

How to run the code:
* run_experiment.py: runs the experiments to obtain solutions (Data/Output//Solutions) and results (Data/Output/Results) both as a .pkl file.
* validate_results.py: takes solutions and results as input, tests the solutions on a test set and saves the results as a validation (Data/Output/Validation)
* analyse_results.py: takes a validation as input, computes statistics (average, standard deviation) and saves it as an analysis (Data/Output/Analysis)
* build_table.py: takes an analysis as input, creates a table (Data/Output/Table) with a format as presented in the article

If you have any questions, feel free to send an email to n.j.schutte@tudelft.nl

