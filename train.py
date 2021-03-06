from blood_model import train
import _pickle as pickle

# Training arguments
dt, dv = train(**{
    "image_dir": "training_dataset",
    "training_iterations": 500,
    "summaries_dir": "tf_files/training_summaries/basic",
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


#####Output used to produce the Train\VAlidation Accuracy report on Tableau
pickle.dump(dt, open('aux_data\\dt.pickle', 'wb'))
pickle.dump(dv, open('aux_data\\dv.pickle', 'wb'))
print (dt)
print (dv)

# dt =  pickle.load(open('aux_data\\dt.pickle' , 'rb'))
# dv =  pickle.load(open('aux_data\\dv.pickle' , 'rb'))

with open("aux_data\\dt.csv", "w") as f:
    for key in dt:
        f.write(str(key) + "," + str(dt[key])+'\n')

with open("aux_data\\dv.csv", "w") as f2:
    for key in dv:
        f2.write(str(key) + "," + str(dv[key])+'\n')
