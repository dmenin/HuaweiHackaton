from blood_model import train
import _pickle as pickle
import pandas as pd
import csv

# Training arguments
<<<<<<< HEAD
train(**{
    "image_dir": "dataset",
    "training_iterations": 500,
    "summaries_dir": "tf_files/training_summaries/basic",
=======
dt, dv = train(**{
    "image_dir": "training_dataset",
    "training_iterations": 100,
    #  "summaries_dir": "tf_files/training_summaries/basic",
>>>>>>> 4925d26d7cf9ee81d8a18bdba62452204da893c3
    "bottleneck_dir": "tf_files/bottlenecks",
    "output_graph": "tf_files/retrained_graph.pb",
    "output_labels": "tf_files/retrained_labels.txt",
    "test_percentage": 10,
    "validation_percentage": 10,
    "learning_rate": 00.1,
    "tensor_name": "final_result",
    "flip":  True,
    # "model_dir": ".",
    "train_batch_size": 100,
    "validation_batch_size": 100,
    "test_batch_size": -1,
    "eval_step_interval": 10,
    "misclassified_print": False,
})


pickle.dump(dt, open('dt.pickle', 'wb'))
pickle.dump(dv, open('dv.pickle', 'wb'))
print (dt)
print (dv)


#dt =  pickle.load(open('dt.pickle' , 'rb'))
#dv =  pickle.load(open('dv.pickle' , 'rb'))

with open("dt.csv", "w") as f:
    for key in dt:
        f.write(str(key) + "," + str(dt[key])+'\n')

with open("dv.csv", "w") as f2:
    for key in dv:
        f2.write(str(key) + "," + str(dv[key])+'\n')