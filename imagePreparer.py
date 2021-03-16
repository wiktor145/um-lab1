import os
import pickle
import random

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import load_img, img_to_array

from model.simnet import simnet

DATASET_SIZE = 30000
RANDOM_SEED = 2137
BATCH_SIZE = 128


class Dataset:
    """
    sequence1 - list that contains 9 lists - each one for one room (from sequence 1)
    each list is a lit of arrays of each photo vector representations (it is [ [[values]], [[values]] ... ]) ; np.array dtype=float32 )
    to get vector: eg. sequence1[3][100][0] -> fourth room, 101 image

    sequence2 - same but for sequence 2

    names - list of names for those 9 rooms (same order as in other lists)
['Corridor1_RGB', 'Corridor2_RGB', 'Corridor3_RGB', 'D3A_RGB', 'D7_RGB', 'F102_RGB', 'F104_RGB', 'F105_RGB', 'F107_RGB']

    queryImages - 9 lists of query images
    each list is for one room
    each list has from 1 to 3 images
    each list is a list of arrays (it is [[values], [values] ...) (not [ [[values]], [[values]] ... ]! ); np.array dtype=float32
    of images vectors representations
    """

    def __init__(self, sequence1, sequence2, query, names):
        self.sequence1 = sequence1
        self.sequence2 = sequence2
        self.queryImages = query
        self.names = names


class ImagePreparer:

    def __init__(self):
        self._perpareVGGNet()
        self.cache_file = "dataset.ds"

    def _perpareVGGNet(self):
        self.vgg = VGG16(include_top=False, pooling="max", input_shape=(480, 640, 3))
        # self.vgg.summary()

    def getVectorFromImage(self, imagePath):
        image = load_img(imagePath)
        image_arr = img_to_array(image)
        image_arr = np.array([image_arr])
        result = self.vgg.predict(image_arr)
        return result[0]

    def getVectorsForImages(self, images_paths):
        images = [load_img(image) for image in images_paths]
        images_arr = [img_to_array(image) for image in images]
        result = []
        for image in images_arr:
            result.append(self.vgg.predict(np.array([image])))
        return result

    def getDataset(self, with_save=False):
        sequence1 = []
        sequence2 = []
        query_images = []

        seq1_path = "./files/DataSet_Nao_RAW/DataSet_SEQUENCE_1"
        seq1_dirs = [item for item in sorted(os.listdir(seq1_path)) if os.path.isdir(os.path.join(seq1_path, item))]

        print("Getting vectors for images from sequence 1")

        for dir in seq1_dirs:
            print(dir)
            images_paths = []
            for f in os.listdir(os.path.join(seq1_path, dir)):
                images_paths.append(os.path.join(seq1_path, dir, f))

            sequence1.append(self.getVectorsForImages(images_paths))

        seq2_path = "./files/DataSet_Nao_RAW/DataSet_SEQUENCE_2"
        seq2_dirs = [item for item in sorted(os.listdir(seq2_path)) if os.path.isdir(os.path.join(seq2_path, item))]

        print("Getting vectors for images from sequence 2")

        for dir in seq2_dirs:
            print(dir)
            images_paths = []
            for f in os.listdir(os.path.join(seq2_path, dir)):
                images_paths.append(os.path.join(seq2_path, dir, f))

            sequence2.append(self.getVectorsForImages(images_paths))

        query_path = "./files/DataSet_Nao_PlaceRecognition/SEQUENCE_2"
        images_for_current = []

        queries_dirs = [item for item in sorted(os.listdir(query_path)) if
                        os.path.isdir(os.path.join(query_path, item))]

        for dir in queries_dirs:
            print(dir)
            image = None
            for f in os.listdir(os.path.join(query_path, dir, "query")):
                if f.endswith(".png"):
                    image = self.getVectorFromImage(os.path.join(query_path, dir, "query", f))

            if dir.endswith("_1") and images_for_current:
                query_images.append(images_for_current)
                images_for_current = []

            images_for_current.append(image)

        query_images.append(images_for_current)

        dataSet = Dataset(sequence1, sequence2, query_images, seq1_dirs)

        if with_save:
            self.save(dataSet)

        return dataSet

    def save(self, dataSet):
        with open(self.cache_file, "wb") as output:
            pickle.dump(dataSet, output, pickle.HIGHEST_PROTOCOL)

    def load_data_from_file(self):
        try:
            f = open(self.cache_file, "rb")
        except:
            return None
        else:
            with f:
                return pickle.load(f)


# #test = ImagePreparer()
# #test.getDataset(with_save=True)
#

def make_tf_dataset(dataset, test_set_size=0.8):
    random.seed(RANDOM_SEED)
    sequence = dataset.sequence1
    index_pairs = []
    for i in range(len(sequence)):
        for j in range(i, len(sequence)):
            index_pairs.append((i, j))

    samples_per_pair = DATASET_SIZE // len(index_pairs)
    examples = []
    labels = []

    for i, j in index_pairs:
        for _ in range(samples_per_pair):
            labels.append(1 if i == j else 0)
            sample = np.zeros((1, 512, 2))
            first = random.choice(sequence[i])
            second = random.choice(sequence[j])
            sample[0, :, 0] = first
            sample[0, :, 1] = second
            examples.append(sample)

    examples = np.asarray(examples)
    labels = np.asarray(labels)

    # split_index = round(test_set_size * examples.shape[0])
    train_examples = examples
    train_labels = labels
    # test_examples = examples[split_index:]
    # test_labels = labels[split_index:]

    train_dataset = tf.data.Dataset.from_tensor_slices((train_examples, train_labels))
    train_dataset = train_dataset.batch(BATCH_SIZE)
    # test_dataset = tf.data.Dataset.from_tensor_slices((test_examples, test_labels))
    # test_dataset = test_dataset.batch(BATCH_SIZE)

    return train_dataset


def test_net(net, ds):
    query_images = ds.queryImages
    seq2 = ds.sequence2

    correct_guesses = [0 for _ in seq2]

    for a, room in enumerate(seq2):
        # print("room ", a)
        for b, image in enumerate(room):
            # if b % 100 == 0:
            # print("image ", b)
            current_guesses = [0 for _ in seq2]

            for i, query_room in enumerate(query_images):
                for query in query_room:
                    # print(i)
                    # print(image[0])
                    sample = np.zeros((1, 512, 2))
                    sample[0, :, 0] = image[0]
                    sample[0, :, 1] = query
                    guess = net.predict(np.expand_dims(sample, axis=0))[0][0][0][0]
                    # print(guess[0][0][0])
                    current_guesses[i] = guess if guess > current_guesses[i] else current_guesses[i]

            guess = np.argmax(current_guesses)
            if a == guess:
                correct_guesses[a] += 1

    for a in range(len(seq2)):
        correct_guesses[a] /= len(seq2[a])

    print("Precisions:")
    for i, a in enumerate(correct_guesses):
        print("room ", i, correct_guesses[i])
    print("Mean average precision:")
    print(np.average(correct_guesses))


if __name__ == '__main__':
    test = ImagePreparer()
    ds = test.load_data_from_file()
    if not ds:
        print("Was not able to load dataset from file dataset.ds")
        print("Trying to process images...")
        ds = test.getDataset(with_save=True)

    train_dataset = make_tf_dataset(ds).shuffle(50)
    net = simnet()
    net.fit(train_dataset, epochs=20, shuffle=True)
    test_net(net, ds)
