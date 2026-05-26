# COMM075 Machine Learning for Data Science
## Group: Gochugaru 



## Group Members


Lubaba  RL Data & Env, Decision Tree Training, SVM Evaluation 
Kim | RL Agent & Training, Naive Bayes Training, MLP Evaluation 
Anas | RL Evaluation, PyGoL Training, Aleph Evaluation 
Long | RL Env Design, Decision Tree Evaluation, SVM Training 
Rahin | RL State Analysis, Naive Bayes Evaluation, MLP Training 
Safayet | RL Baseline & Results, PyGoL Evaluation, Aleph Training 



## Project Structure

COMM075_Gochugaru/
├── README.md                        ← you are here
├── thread1_HAR/                     ← UCI HAR Dataset
│ 
│   ├── EDA_HAR.ipynb
│   ├── DecisionTree_HAR.ipynb
│   ├── NaiveBayes_HAR.ipynb
│   ├── PyGoL_HAR.ipynb
│   ├── Comparison_HAR.ipynb
│   ├── ucihar_train.csv
│   ├── ucihar_test.csv
│   └── outputs/
│       ├── dt_har/
│       ├── nb_har/
│       ├── pygol_har/
│       └── comparison_har/
├── thread2_phiusiil/                ← PhiUSIIL Phishing Dataset
│ 
│   ├── 01_eda_phiusiil.ipynb
│   ├── 02_aleph_phiusiil.ipynb
│   ├── 03_svm_rbf_phiusiil.ipynb
│   ├── 04_mlp_phiusiil.ipynb
│   ├── 05_evaluate_thread2_phiusiil.ipynb
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv
│   ├── aleph.pl
│   ├── aleph_files/
│   └── outputs/
│       ├── aleph_phiusiil/
│       ├── svm_phiusiil/
│       ├── mlp_phiusiil/
│       └── thread2_comparison/
└── thread3_RL/                      ← Yahoo Finance Stock Trading
    
    ├── yahoo_rl.ipynb
    ├── yahoo_data.py
    ├── yahoo_env.py
    ├── yahoo_qagent.py
    ├── yahoo_evo.py
    ├── yahoo_evF.py
    ├── yahoo_test.py
    ├── data_full.csv
    ├── data_train.csv
    ├── data_test.csv
    └── outputs/
        ├── rl_eda/
        ├── rl_environment/
        ├── rl_analysis/
        ├── rl_training/
        ├── rl_evaluation/
        └── rl_results/



## Execution Order

Run notebooks in this order to avoid dependency errors:


1.  yahoo_data.py                          
2.  yahoo_evo.py                           
3.  yahoo_evF.py                           
4.  yahoo_test.py                          
5.  yahoo_rl.ipynb                         
6.  EDA_HAR.ipynb                          
7.  DecisionTree_HAR.ipynb                 
8.  NaiveBayes_HAR.ipynb                   
9.  PyGoL_HAR.ipynb                        
10. Comparison_HAR.ipynb                   
11. 01_eda_phiusiil.ipynb                  
12. 02_aleph_phiusiil.ipynb                
13. 03_svm_rbf_phiusiil.ipynb              
14. 04_mlp_phiusiil.ipynb                  
15. 05_evaluate_thread2_phiusiil.ipynb     


