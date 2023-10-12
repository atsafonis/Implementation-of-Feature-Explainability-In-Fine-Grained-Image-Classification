import os
import torch
import numpy as np
from statistics import mean 

predicted_att_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/results_con5b_lay/best_results_312.txt'
ground_truth_path = '/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/human_most_important_att'

predicted_att = np.loadtxt(predicted_att_path, dtype=str)

avg_map_score = 0
total_MAP_score = 0

for i in range(len(predicted_att)):
    #print(predicted_att[i]) 
    total_avg_precision = 0
    split_predicted = predicted_att[i].split("/")
    x = split_predicted[0].split('_0')  #I want the name of the bird class only 
    class_name = x[0]
    #print(class_name)
    
    ground_truth_path = f'/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/human_most_important_att/{class_name}.txt'
    ground_truth = np.loadtxt(ground_truth_path, dtype=str, usecols = (0))
    #print(ground_truth)

    true_positives = 0
    avg_precision = 0


    for j in range(1,len(split_predicted)):
        if (split_predicted[j] in ground_truth):
            for a in range(1, j):
                if (split_predicted[a] in ground_truth):
                    true_positives += 1
                    
        precision = true_positives/j  
        true_positives = 0
        avg_precision += precision
    
    
    total_avg_precision = avg_precision/len(ground_truth)
    avg_map_score += total_avg_precision
    
    

print("Total MAP score for all the test images is:")
total_MAP_score = avg_map_score/len(predicted_att)


print("{:.0%}".format(total_MAP_score))