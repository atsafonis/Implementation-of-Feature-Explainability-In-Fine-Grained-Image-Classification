import glob
import numpy as np

#results =  np.loadtxt('/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/results_con5b_lay/time_results2.txt', dtype=str)

results = open("/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/results_con5b_lay/time_results2.txt", "r")


lines = results.read()
list_of_results = lines.splitlines()

results.close()

split_list = list(range(len(list_of_results)))
counter1 = 0
counter2 = 0
counter3 = 0
counter4 = 0
#counter5 = 0
for i in range(len(list_of_results)):
    #print(list_of_results[i]) 
    split_list = list_of_results[i].split("#")

    for j in range(len(split_list)):
        x = split_list[0].split('_0')  #I want the name of the bird class only 
        class_name = x[0]
        ground_truth_path = f'/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/human_most_important_att/{class_name}.txt'
        ground_truth = np.loadtxt(ground_truth_path, dtype=str, usecols = (0))


        if(ground_truth[0] == split_list[j]):
            counter1 += j

        if(ground_truth[1] == split_list[j]):
            counter2 += j

        if(ground_truth[2] == split_list[j]):
            counter3 += j

        if(ground_truth[3] == split_list[j]):
            counter4 += j

        #if(ground_truth[4] == split_list[j]):
            #counter5 += j


avg_position1 = counter1/len(list_of_results)
avg_position2 = counter2/len(list_of_results)
avg_position3 = counter3/len(list_of_results)
avg_position4 = counter4/len(list_of_results)
#avg_position5 = counter5/len(list_of_results)

print(avg_position1)
print(avg_position2)
print(avg_position3)
print(avg_position4)
#print(avg_position5)




    #print(split_predicted)
    #x = split_predicted[0].split('_0')  #I want the name of the bird class only 
    #class_name = x[0]
    #print(class_name)
    
    #ground_truth_path = f'/nfs/bigcortex/atsafonis/cub_classification/MMAL-Net/human_most_important_att/{class_name}.txt'
    #ground_truth = np.loadtxt(ground_truth_path, dtype=str, usecols = (0))
    #print(ground_truth)
    #print(ground_truth[0])

    














