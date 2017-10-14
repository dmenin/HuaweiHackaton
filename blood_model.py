# Import OS
import os
# Import Regular expression
import re
# Import Random
import random
# Import datetime
from datetime import datetime
# Import numpy as np
import numpy as np
# Import Hashlib
import hashlib

# Import TensorFlow
import tensorflow as tf
# Import gFile
from tensorflow.python.platform import gfile
# Import Graph Util
from tensorflow.python.framework import graph_util

# Set the environ
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BOTTLENECK_TENSOR_SIZE = 2048
BOTTLENECK_TENSOR_NAME = 'pool_3/_reshape:0'
RESIZED_INPUT_TENSOR_NAME = 'ResizeBilinear:0'
JPEG_DATA_TENSOR_NAME = 'DecodeJpeg/contents:0'
MAX_NUM_IMAGES_PER_CLASS = 2 ** 27 - 1  # ~134M


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


# Function to ensure directories exist
def ensure_dir_exists(dir_name):
    # If it doesn't exist
    if not os.path.exists(dir_name):
        # Create it
        os.makedirs(dir_name)


def load_files(image_dir, testing_percentage, validation_percentage):
    # Check whether the image directory exists
    if not gfile.Exists(image_dir):
        # Print error message and return
        print("The image directory '" + image_dir + "' not found.")
        return None

    # Valid list of image extensions
    extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
    # Empty container to hold the found images
    result = {}

    # Iterate over the subdirectories
    for sub_dir in [x[0] for x in gfile.Walk(image_dir)][1:]:
        # Get current subdirectory
        dir_name = os.path.basename(sub_dir)
        # Print status message
        print("Loading '" + dir_name + "'")
        # Empty container to hold the files
        file_list = []
        # For each extension in the list
        for ext in extensions:
            file_glob = os.path.join(image_dir, dir_name, '*.' + ext)
            # Append the files to the file list
            file_list.extend(gfile.Glob(file_glob))
        # Print message if folder is empty or doesn't have the right filetypes
        if not file_list:
            print('No files found')

        # Print message if file list is not long enough
        if len(file_list) < 20:
            print('Folder has less than 20 images. May be a problem.')

        else:
            print("Found " + str(len(file_list)) + " images.")
        # Sanitize label
        label_name = re.sub(r'[^a-z0-9]+', ' ', dir_name.lower())

        # Empty lists to hold training, testining and validation images
        training_images = []
        testing_images = []
        validation_images = []

        # Iterate through the file list
        for file_name in file_list:
            # Grab the base file name
            base_name = os.path.basename(file_name)
            # Get a hash of the file name
            hash_name = re.sub(r'_nohash_.*$', '', file_name)
            # Calculate the hash
            hash_name_hashed = hashlib.sha1(
                hash_name.encode("utf-8")).hexdigest()
            percentage_hash = ((int(hash_name_hashed, 16) %
                                (MAX_NUM_IMAGES_PER_CLASS + 1)) *
                               (100.0 / MAX_NUM_IMAGES_PER_CLASS))
            # Use some files for validation
            if percentage_hash < validation_percentage:
                validation_images.append(base_name)
            # Use some files for testing
            elif percentage_hash < (testing_percentage +
                                    validation_percentage):
                testing_images.append(base_name)
            # Use remaining files for training
            else:
                training_images.append(base_name)

        # Populate the result dictionary
        result[label_name] = {
            'dir': dir_name,
            'training': training_images,
            'testing': testing_images,
            'validation': validation_images,
        }

    # Return result
    return result


# Function that caches bottlenecks
def cache_bottleneck(session, image_list, image_dir, bottleneck_dir,
                     jpeg_data_tensor, bottleneck_tensor):
    # Number of bottlenecks
    bottlenecks = 0
    # Ensure directories existr
    ensure_dir_exists(bottleneck_dir)
    # For every item in thge image list
    for label_name, label_lists in image_list.items():
        # For each category
        for category in ['training', 'testing', 'validation']:
            # Get the image of the given category
            category_list = label_lists[category]
            # For each item in the category list
            for index, unused_base_name in enumerate(category_list):
                # Create bottleneck for each category
                create_bottleneck(session, image_list, label_name, index,
                                  image_dir, category, bottleneck_dir,
                                  jpeg_data_tensor, bottleneck_tensor)
                # Increment the bottleneck
                bottlenecks += 1
                if bottlenecks % 100 == 0:
                    print(str(bottlenecks) + ' bottleneck files created.')


# Function that creates bottlenecks
def create_bottleneck(session, image_list, label_name, index, image_dir,
                      category, bottleneck_dir, jpeg_data_tensor,
                      bottleneck_tensor):

    # Get a given category
    label_lists = image_list[label_name]
    # Get its directory
    sub_dir = label_lists['dir']
    # Construct its full path
    sub_dir_path = os.path.join(bottleneck_dir, sub_dir)
    # Ensure the directory exists
    ensure_dir_exists(sub_dir_path)
    # Get the bottleneck file path
    bottleneck_path = get_bottleneck_path(image_list, label_name, index,
                                          bottleneck_dir, category)
    # If there isn't such a path
    if not os.path.exists(bottleneck_path):
        # Create the bottleneck file
        create_bottleneck_file(bottleneck_path, image_list, label_name, index,
                               image_dir, category, session, jpeg_data_tensor,
                               bottleneck_tensor)
    # Open the bottleneck file
    with open(bottleneck_path, 'r') as bottleneck_file:
        # Read from the bottleneck file
        bottleneck_string = bottleneck_file.read()
    did_hit_error = False
    # Check if the values are corrupted
    try:
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # If there's a value error
    except ValueError:
        # Output error message
        print('Invalid float found, recreating bottleneck')
        did_hit_error = True

    if did_hit_error:
        # Recreate bottleneck file
        create_bottleneck_file(bottleneck_path, image_list, label_name, index,
                               image_dir, category, session, jpeg_data_tensor,
                               bottleneck_tensor)

        # Reload the file
        with open(bottleneck_path, 'r') as bottleneck_file:
            bottleneck_string = bottleneck_file.read()

        # Get the bottleneck values
        bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
    # Return bottleneck values
    return bottleneck_values


# Function to create a bottlneck file
def create_bottleneck_file(bottleneck_path, image_list, label_name, index,
                           image_dir, category, session, jpeg_data_tensor,
                           bottleneck_tensor):
    # Output bottleneck creation
    print('Creating bottleneck at ' + bottleneck_path)
    # Get the image path
    image_path = get_image_path(image_list, label_name, index,
                                image_dir, category)
    # If the file does not exist
    if not gfile.Exists(image_path):
        # Output fatal error
        tf.logging.fatal('File does not exist %s', image_path)

    # Otherwise, read from the file
    image_data = gfile.FastGFile(image_path, 'rb').read()
    # Try to run the bottleneck on an image
    try:
        bottleneck_values = run_bottleneck_on_image(
            session,
            image_data,
            jpeg_data_tensor,
            bottleneck_tensor)
    # In case of error
    except:
        # Raise error
        raise RuntimeError('Error during processing file %s' % image_path)

    # Create a string with the bottleneck values
    bottleneck_string = ','.join(str(x) for x in bottleneck_values)
    # Open the bottleneck file
    with open(bottleneck_path, 'w') as bottleneck_file:
        # Write the bottleneck file
        bottleneck_file.write(bottleneck_string)


# Function to extract the summary layer
def run_bottleneck_on_image(session, image_data, image_data_tensor,
                            bottleneck_tensor):

    # Obtain the bottleneck values from the session
    bottleneck_values = session.run(
        bottleneck_tensor,
        {image_data_tensor: image_data})

    # Return squeezed the values
    return np.squeeze(bottleneck_values)


# Function to get the bottleneck path
def get_bottleneck_path(image_list, label_name, index, bottleneck_dir,
                        category):
    # Return the image path
    return get_image_path(image_list, label_name, index, bottleneck_dir,
                          category) + '.txt'


# Function to get thge image path
def get_image_path(image_list, label_name, index, image_dir, category):
    # If label name doesn't exist
    if label_name not in image_list:
        # Throw error
        tf.logging.fatal('Label does not exist %s.', label_name)
    # Get list of files per label
    label_lists = image_list[label_name]
    # If category doesn't exist
    if category not in label_lists:
        # Throw error
        tf.logging.fatal('Category does not exist %s.', category)
    # Get category of files
    category_list = label_lists[category]
    # Check whether it's empty
    if not category_list:
        # Throw error
        tf.logging.fatal('Label %s has no images in the category %s.',
                         label_name, category)
    # Get the index
    mod_index = index % len(category_list)
    # Get its base name
    base_name = category_list[mod_index]
    # Get its directory
    sub_dir = label_lists['dir']
    # Construct and return the full path
    return os.path.join(image_dir, sub_dir, base_name)


# Function to retrieve random cached bottlenecks
def get_random_cached_bottlenecks(session, image_list, batch_size, category,
                                  bottleneck_dir, image_dir, jpeg_data_tensor,
                                  bottleneck_tensor):

    # Number of images in this given class
    class_number = len(image_list)
    # Empty lists to hold the bottlenecks
    bottlenecks = []
    ground_truths = []
    filenames = []

    # If the batch size is larger than zero
    if batch_size >= 0:
        # Retrieve a random sample of bottlenecks
        for unused_i in range(batch_size):
            # Get a random label index
            label_index = random.randrange(class_number)
            # Get the associated name
            label_name = list(image_list)[label_index]
            # Get a random image index
            image_index = random.randrange(MAX_NUM_IMAGES_PER_CLASS + 1)
            # Get the image name
            image_name = get_image_path(
                image_list, label_name, image_index, image_dir, category)

            # Get the given bottleneck
            bottleneck = create_bottleneck(
                session,
                image_list,
                label_name,
                image_index,
                image_dir,
                category,
                bottleneck_dir,
                jpeg_data_tensor,
                bottleneck_tensor)

            # Create an array filled with zeros
            ground_truth = np.zeros(class_number, dtype=np.float32)
            ground_truth[label_index] = 1.0
            # Append the given bottleneck
            bottlenecks.append(bottleneck)
            ground_truths.append(ground_truth)
            # Append image name
            filenames.append(image_name)

    # Otherwise
    else:
        # Retrieve all bottlenecks
        for label_index, label_name in enumerate(image_list):
            # For each index and label
            for image_index, image_name in enumerate(
                    image_list[label_name][category]):
                # Get the image image path
                image_name = get_image_path(
                    image_list, label_name, image_index, image_dir, category)
                # Get the given bottleneck
                bottleneck = create_bottleneck(
                    session,
                    image_list,
                    label_name,
                    image_index,
                    image_dir,
                    category,
                    bottleneck_dir,
                    jpeg_data_tensor,
                    bottleneck_tensor)

                # Create an array filled with zeros
                ground_truth = np.zeros(class_number, dtype=np.float32)
                ground_truth[label_index] = 1.0
                # Append the bottleneck
                bottlenecks.append(bottleneck)
                ground_truths.append(ground_truth)
                # Append the image name
                filenames.append(image_name)

    # Return bottlenecks, ground truths and the filenames
    return bottlenecks, ground_truths, filenames


# Function to do final training operations
def final_training_ops(class_count, final_tensor_name, bottleneck_tensor,
                       learning_rate):
    # With the input
    with tf.name_scope('input'):
        bottleneck_input = tf.placeholder_with_default(
            bottleneck_tensor,
            shape=[None, BOTTLENECK_TENSOR_SIZE],
            name='BottleneckInputPlaceholder')

        ground_truth_input = tf.placeholder(
            tf.float32,
            [None, class_count],
            name='GroundTruthInput')

    # Label the last layer
    layer_name = 'final_layer'
    # With the last layer context
    with tf.name_scope(layer_name):
        # With its weigths context
        with tf.name_scope('weights'):
            # Initilizae the weights
            initial_value = tf.truncated_normal(
                [BOTTLENECK_TENSOR_SIZE, class_count],
                stddev=0.001)

            # Set the final weights
            layer_weights = tf.Variable(initial_value, name='final_weights')

            variable_summaries(layer_weights)

        with tf.name_scope('biases'):
            layer_biases = tf.Variable(tf.zeros([class_count]),
                                       name='final_biases')
            variable_summaries(layer_biases)

        with tf.name_scope('Wx_plus_b'):
            logits = tf.matmul(bottleneck_input, layer_weights) + layer_biases
            tf.summary.histogram('pre_activations', logits)

    final_tensor = tf.nn.softmax(logits, name=final_tensor_name)
    tf.summary.histogram('activations', final_tensor)

    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=ground_truth_input,
            logits=logits)

        with tf.name_scope('total'):
            cross_entropy_mean = tf.reduce_mean(cross_entropy)

    tf.summary.scalar('cross_entropy', cross_entropy_mean)
    with tf.name_scope('train'):
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
        train_step = optimizer.minimize(cross_entropy_mean)

    return (train_step, cross_entropy_mean, bottleneck_input,
            ground_truth_input, final_tensor)


# Function to insert operation to evaluate the final layer
def evaluation_step(result_tensor, ground_truth_tensor):
    # With the accuracy context
    with tf.name_scope('accuracy'):
        # With the correct prediction context
        with tf.name_scope('correct_prediction'):
            # Make a prediction
            prediction = tf.argmax(result_tensor, 1)
            # Calculate the correct prediction
            correct_prediction = tf.equal(
                prediction, tf.argmax(ground_truth_tensor, 1))

        # With the accuracy context
        with tf.name_scope('accuracy'):
            # Calculate the evaluation step
            eval_step = tf.reduce_mean(tf.cast(
                correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', eval_step)
    # Return the evaluation and the prediction
    return eval_step, prediction


# Most awesome function that changes the world
def train(**kwargs):
    # Setup the directory we'll write summaries to for TensorBoard
    if tf.gfile.Exists(kwargs.get("summaries_dir")):
        tf.gfile.DeleteRecursively(kwargs.get("summaries_dir"))
    tf.gfile.MakeDirs(kwargs.get("summaries_dir"))

    # Load the pre-created graph
    with tf.Graph().as_default() as graph:
        # Construct the filename
        model_filename = os.path.join(
            kwargs.get("model_dir", "."),
            'classify_image_graph_def.pb')

        # Open the graph definition file
        with gfile.FastGFile(model_filename, 'rb') as f:
            # Create a new graph instance
            graph_definition = tf.GraphDef()
            # Load the pre-created graph
            graph_definition.ParseFromString(f.read())
            # Import graph definition
            bottleneck_tensor, jpeg_data_tensor, resized_input_tensor = (
                tf.import_graph_def(
                    graph_definition,
                    name='',
                    return_elements=[BOTTLENECK_TENSOR_NAME,
                                     JPEG_DATA_TENSOR_NAME,
                                     RESIZED_INPUT_TENSOR_NAME]))

    # Load image files
    image_list = load_files(kwargs.get("image_dir", "dataset"),
                            kwargs.get("testing_percentage", 10),
                            kwargs.get("validation_percentage", 10))

    # Calculate the number of classes
    class_number = len(image_list)
    # If no classes were found
    if class_number == 0:
        print('No valid directories were found at ' + kwargs.get("image_dir",
                                                                 "dataset"))
        return -1
    # If a single class was found
    if class_number == 1:
        print('Only a single class at ' + kwargs.get("image_dir", "dataset"))
        return -1

    # Start the training session
    with tf.Session(graph=graph) as session:
        # We'll make sure we've calculated the 'bottleneck' image summaries and
        # cached them on disk.
        cache_bottleneck(
            session,
            image_list,
            kwargs.get("image_dir", "dataset"),
            kwargs.get("bottleneck_dir", "tf_files/bottlenecks"),
            jpeg_data_tensor,
            bottleneck_tensor)

        # Add a new layer -- which will be trained later.
        (train_step,
         cross_entropy,
         bottleneck_input,
         ground_truth_input,
         final_tensor) = final_training_ops(
             len(image_list),
             kwargs.get("tensor_name", "final_result"),
             bottleneck_tensor,
             kwargs.get("learning_rate", 0.01))

        # Create the operations to evaluate the accuracy of the new layer.
        eval_step, prediction = evaluation_step(final_tensor,
                                                ground_truth_input)

        # Merge all the summaries and write them out to the summaries_dir
        merged = tf.summary.merge_all()
        # Output the training outcome
        train_writer = tf.summary.FileWriter(
            "/tmp/retrain_logs" + '/train', session.graph)

        # Output the training outcome
        validation_writer = tf.summary.FileWriter(
            "/tmp/retrain_logs" + '/validation')

        # Set up all our weights to their initial default values.
        init = tf.global_variables_initializer()
        session.run(init)

        iterations = kwargs.get("training_iterations", 500)
        # Run the training for this mani iterations.
        for i in range(iterations):
            # Get a batch of input bottleneck values.
            (train_bottlenecks, train_ground_truth, _) = \
                    get_random_cached_bottlenecks(
                 session,
                 image_list,
                 kwargs.get("train_batch_size"),
                 'training',
                 kwargs.get("bottleneck_dir"),
                 kwargs.get("image_dir"),
                 jpeg_data_tensor,
                 bottleneck_tensor)

            # Feed bottlenecks and ground truth into the graph and run training
            train_summary, _ = session.run(
                [merged, train_step],
                feed_dict={bottleneck_input: train_bottlenecks,
                           ground_truth_input: train_ground_truth})
            train_writer.add_summary(train_summary, i)

            is_last_step = (i + 1 == iterations)
            if (i % kwargs.get("eval_step_interval")) == 0 or is_last_step:
                train_accuracy, cross_entropy_value = session.run(
                    [eval_step, cross_entropy],
                    feed_dict={bottleneck_input: train_bottlenecks,
                               ground_truth_input: train_ground_truth})

                print('%s: Step %d: Train accuracy = %.1f%%' % (
                    datetime.now(), i, train_accuracy * 100))
                print('%s: Step %d: Cross entropy = %f' % (
                    datetime.now(), i, cross_entropy_value))

                validation_bottlenecks, validation_ground_truth, _ = (
                    get_random_cached_bottlenecks(
                        session, image_list,
                        kwargs.get("validation_batch_size"),
                        'validation',
                        kwargs.get("bottleneck_dir"),
                        kwargs.get("image_dir"), jpeg_data_tensor,
                        bottleneck_tensor))

                # Run a validation step and capture training summaries
                validation_summary, validation_accuracy = session.run(
                    [merged, eval_step],
                    feed_dict={bottleneck_input: validation_bottlenecks,
                               ground_truth_input: validation_ground_truth})

                validation_writer.add_summary(validation_summary, i)
                print('%s: Step %d: Validation accuracy = %.1f%% (N=%d)' %
                      (datetime.now(), i, validation_accuracy * 100,
                       len(validation_bottlenecks)))

        # We've completed all our training, so run a final test evaluation
        test_bottlenecks, test_ground_truth, test_filenames = (
            get_random_cached_bottlenecks(
                session,
                image_list,
                kwargs.get("test_batch_size"),
                'testing',
                kwargs.get("bottleneck_dir"),
                kwargs.get("image_dir"),
                jpeg_data_tensor,
                bottleneck_tensor))

        test_accuracy, predictions = session.run(
            [eval_step, prediction],
            feed_dict={bottleneck_input: test_bottlenecks,
                       ground_truth_input: test_ground_truth})

        print('Final test accuracy = %.1f%% (N=%d)' % (
            test_accuracy * 100, len(test_bottlenecks)))

        if kwargs.get("misclassified_print"):
            print('=== MISCLASSIFIED TEST IMAGES ===')
            for i, test_filename in enumerate(test_filenames):
                if predictions[i] != test_ground_truth[i].argmax():
                    print('%70s  %s' % (
                        test_filename,
                        list(image_list)[predictions[i]]))

        # Write out the trained graph and labels with the weights stored as
        # constants.
        output_graph_def = graph_util.convert_variables_to_constants(
            session,
            graph.as_graph_def(),
            [kwargs.get("tensor_name", "final_result")])

        with gfile.FastGFile(kwargs.get("output_graph"), 'wb') as f:
            f.write(output_graph_def.SerializeToString())

        with gfile.FastGFile(kwargs.get("output_labels"), 'w') as f:
            f.write('\n'.join(image_list.keys()) + '\n')
