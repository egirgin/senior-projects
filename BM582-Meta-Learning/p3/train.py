import numpy as np
import random
import tensorflow as tf
from tensorflow.keras import layers
from load_data import *
import argparse

arg_parser = argparse.ArgumentParser()

arg_parser.add_argument("-k", "--num_samples", type=int)
arg_parser.add_argument("-n", "--num_classes", type=int)
arg_parser.add_argument("-b", "--batch_size", type=int)
arg_parser.add_argument("-lr", "--learning_rate", type=float, default=0.001)
arg_parser.add_argument("-c", "--use_conv", action="store_true")
arg_parser.add_argument("-i", "--iteration", type=int, default=2500)

args = arg_parser.parse_args()

class MANN(tf.keras.Model):

    def __init__(self, num_classes, samples_per_class):
        super(MANN, self).__init__()
        self.num_classes = num_classes
        self.samples_per_class = samples_per_class
        self.layer1 = tf.keras.layers.LSTM(128, return_sequences=True)
        self.layer2 = tf.keras.layers.LSTM(num_classes, return_sequences=True)
        self.conv = tf.keras.layers.Conv2D( filters = 11, 
                                            kernel_size = 3,
                                            padding = "same",
                                            activation = "relu")
        self.conv_out = tf.keras.layers.Conv2D( filters = 1, 
                                            kernel_size = 3,
                                            padding = "same",
                                            activation = "relu")

    def call(self, input_images, input_labels):
        """
        MANN
        Args:
            input_images: [B, K+1, N, 784] flattened images
            labels: [B, K+1, N, N] ground truth labels
        Returns:
            [B, K+1, N, N] predictions
        """
        #############################
        #### YOUR CODE GOES HERE ####
        
        B, K, N, flat = input_images.shape
        
        if args.use_conv:
            input_images = tf.reshape(input_images, (-1, 28, 28, 1))
            input_images = self.conv(input_images)
            input_images = self.conv_out(input_images)
            input_images = tf.reshape(input_images, (B, K*N, -1))

        concat_images = tf.reshape(input_images, (B, K*N, flat)) # Concat images 

        
        # Concat Labels (put zero at the last of K to predict)
        concat_labels = tf.concat(
            (input_labels[:, :K-1, :, :], tf.zeros_like(input_labels[:, K-1:, :, :])),
            axis=1
        )
        
    
        merged_labels = tf.reshape(concat_labels, (B, K*N, N))
        concat_images = tf.concat((concat_images, merged_labels), -1)


        out = self.layer1(concat_images)
        out = self.layer2(out)
        out = tf.reshape(out, (B, K, N, N))
        

        #############################
        return out

    def loss_function(self, preds, labels):
        """
        Computes MANN loss
        Args:
            preds: [B, K+1, N, N] network output
            labels: [B, K+1, N, N] labels
        Returns:
            scalar loss
        """
        #############################
        #### YOUR CODE GOES HERE ####
        return tf.reduce_sum(tf.losses.categorical_crossentropy(labels[:, -1, :, :], preds[:, -1, :, :], from_logits = True))
        #############################


@tf.function
def train_step(images, labels, model, optim, eval=False):
    with tf.GradientTape() as tape:
        predictions = model(images, labels)
        loss = model.loss_function(predictions, labels)
    if not eval:
        gradients = tape.gradient(loss, model.trainable_variables)
        optim.apply_gradients(zip(gradients, model.trainable_variables))
    return predictions, loss


def main(num_classes=5, num_samples=3, meta_batch_size=1, random_seed=1234, learning_rate=0.001):
    random.seed(random_seed)
    np.random.seed(random_seed)
    tf.random.set_seed(random_seed)

    data_generator = DataGenerator(num_classes, num_samples + 1)

    o = MANN(num_classes, num_samples + 1)
    optim = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    log = []

    for step in range(args.iteration):
        i, l = data_generator.sample_batch('train', meta_batch_size)
        _, ls = train_step(i, l, o, optim)

        if (step + 1) % 100 == 0:
            print("*" * 5 + "Iter " + str(step + 1) + "*" * 5)
            i, l = data_generator.sample_batch('test', 100)
            pred, tls = train_step(i, l, o, optim, eval=True)
            print("Train Loss:", ls.numpy(), "Test Loss:", tls.numpy())
            log.append("TrainLoss:" + str(ls.numpy()) +  " TestLoss:" + str(tls.numpy()))
            pred = tf.reshape(pred, [-1, num_samples + 1, num_classes, num_classes])
            pred = tf.math.argmax(pred[:, -1, :, :], axis=2)
            l = tf.math.argmax(l[:, -1, :, :], axis=2)
            print("Test Accuracy", tf.reduce_mean(tf.cast(tf.math.equal(pred, l), tf.float32)).numpy())
            test_acc = float(tf.reduce_mean(tf.cast(tf.math.equal(pred, l), tf.float32)).numpy())
            log.append("TestAccuracy " + str(test_acc))
            if test_acc >= 0.60:
                break



    with open("./c{}_s{}_b{}_lr{}.txt".format(num_classes, num_samples, meta_batch_size, learning_rate), "w+") as logfile:
        logfile.write("\n".join(log))

if __name__=="__main__":
    main(num_classes=args.num_classes, num_samples=args.num_samples, meta_batch_size=args.batch_size, learning_rate=args.learning_rate, random_seed=42)
