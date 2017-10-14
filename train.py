from blood_model import train

# Training arguments
train(**{
    "image_dir": "dataset",
    "training_iterations": 500,
    #  "summaries_dir": "tf_files/training_summaries/basic",
    "bottleneck_dir": "tf_files/bottlenecks",
    "output_graph": "tf_files/retrained_graph.pb",
    "output_labels": "tf_files/retrained_labels.txt",
    "test_percentage": 10,
    "validation_percentage": 10,
    "learning_rate": 0.01,
    "tensor_name": "final_result",
    "flip":  True,
    # "model_dir": ".",
    "train_batch_size": 100,
    "validation_batch_size": 100,
    "test_batch_size": -1,
    "eval_step_interval": 10,
    "misclassified_print": False,
})
