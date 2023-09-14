import os
import torch
import numpy as np
from statistics import mean 

predicted_att_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/results_con5b_lay/results.txt'
ground_truth_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/human_most_important_att'

predicted_att = np.loadtxt(predicted_att_path, dtype=str)

total_avg_precision = 0

for i in range(len(predicted_att)):
    print(predicted_att[i]) 
    split_predicted = predicted_att[i].split("/")
    x = split_predicted[0].split('_0')  #I want the name of the bird class only 
    class_name = x[0]
    print(class_name)
    
    ground_truth_path = f'/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/human_most_important_att/{class_name}.txt'
    ground_truth = np.loadtxt(ground_truth_path, dtype=str, usecols = (0))
    print(ground_truth)

    true_positives = 0
    precision = np.empty(len(split_predicted) - 1)

    for j in range(1,len(split_predicted)):
        for k in range(len(ground_truth)):
            if (split_predicted[j] == ground_truth[k]):
                true_positives += 1

        precision[j-1] = true_positives/(len(split_predicted) - 1)   
        true_positives = 0
    print(precision)
    avg_precision = mean(precision)
    total_avg_precision += avg_precision
    print(avg_precision)

print("Total MAP score for all the test images is:")
MAP_score = total_avg_precision/len(predicted_att)
print(MAP_score)