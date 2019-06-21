from Preprocessor import  Preprocessor
import numpy as np
import tensorflow as tf


class Model(object):

    def __init__(self):
        self.preProcessor = Preprocessor()
        self.lstmSize = 256
        self.lstmLayers = 1
        self.batchSize = 500
        self.learningRate = 0.0005
        self.seqL = 100


    def splitTrainData(self):
        splitFrac = 0.9
        splitIdx = int(len(self.intData) * splitFrac)
        trainX, valX = self.intData[:splitIdx], self.intData[splitIdx:]
        trainY, valY = self.allLabel[:splitIdx], self.allLabel[splitIdx:]

        testIdx = int(len(valX) * 0.5)
        valX, testX = valX[:testIdx], valX[testIdx:]
        valY, testY = valY[:testIdx], valY[testIdx:]

        self.trainX = trainX
        self.trainY = trainY
        self.testX = testX
        self.testY = testY
        self.valX = valX
        self.valY = valY
        print('trainX:{}, trainY:{}'.format(self.trainX.shape, self.trainY.shape))
        print('testX:{}, testY:{}'.format(self.testX.shape, self.testY.shape))
        print('valX:{}, valY:{}'.format(self.valX.shape, self.valY.shape))

    def getBatches(self, x, y, batchSize = 100):
        nBatches = len(x) // batchSize
        x, y = x[:nBatches * batchSize], y[:nBatches * batchSize]
        for ii in range(0, len(x), batchSize):
            yield x[ii:ii + batchSize], y[ii:ii + batchSize]

    def getCell(self):
        return tf.contrib.rnn.BasicLSTMCell(self.lstmSize)

    def bulidModel(self, batchSize, needFinalSigmoid=False):
        nWords = len(self.preProcessor.intToVocab) + 500
        embedSize = 300
        graph = tf.Graph()
        with graph.as_default():
            inputs_ = tf.placeholder(tf.int32, [None, None], name = 'inputs')
            labels_ = tf.placeholder(tf.int32, [None, None], name = 'labels')
            keepProb = tf.placeholder(tf.float32, name = 'keep_prob')

            embedding = tf.Variable(tf.random_uniform((nWords, embedSize), -1, 1))
            embed = tf.nn.embedding_lookup(embedding, inputs_)

            cells = [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.BasicLSTMCell(self.lstmSize), output_keep_prob = keepProb)
                     for _ in range(self.lstmLayers)]
            cell = tf.contrib.rnn.MultiRNNCell(cells)

            # Getting an initial state of all zeros
            initialState = cell.zero_state(batchSize, tf.float32)

            outputs, finalState = tf.nn.dynamic_rnn(cell, embed,
                                                     initial_state = initialState)

            predictions = tf.contrib.layers.fully_connected(outputs[:, -1], 1)
            if needFinalSigmoid:
                predictions = tf.sigmoid(predictions)

            cost = tf.losses.mean_squared_error(labels_, predictions)
            optimizer = tf.train.AdamOptimizer(self.learningRate).minimize(cost)

            return graph, inputs_, labels_, keepProb, cell, initialState, \
                   finalState, cost, optimizer, predictions

    def buildModel2Category(self, batchSize):
        graph, inputs_, labels_, keepProb, cell, \
        initialState, finalState, cost, optimizer, \
        predictions = self.bulidModel(batchSize, True)
        with graph.as_default():
            correctPred = tf.equal(tf.cast(tf.round(predictions), tf.int32), labels_)
            accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

        return graph, inputs_, labels_, keepProb, cell, initialState, \
               finalState, cost, optimizer, predictions, accuracy

    def train(self, epochs):
        self.intData, self.allLabel = self.preProcessor.loadPreprocessedData(self.seqL)
        self.splitTrainData()
        graph, inputs_, labels_, \
        keepProb, cell, initialState, \
        finalState, cost, optimizer, predictions = self.bulidModel(self.batchSize, False)

        with graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph = graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            lossHistory = []
            valLossHistory = []
            for e in range(epochs):
                state = sess.run(initialState)
                for ii, (x, y) in enumerate(self.getBatches(self.trainX, self.trainY, self.batchSize), 1):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keepProb: 0.5,
                            initialState: state}
                    loss, state, _ = sess.run([cost, finalState, optimizer], feed_dict = feed)
                    lossHistory.append(loss)
                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))

                    if iteration % 25 == 0:
                        valState = sess.run(cell.zero_state(self.batchSize, tf.float32))
                        valAllLoss = []
                        for x, y in self.getBatches(self.valX, self.valY, self.batchSize):
                            feed = {inputs_: x,
                                    labels_: y[:, None],
                                    keepProb: 1,
                                    initialState: valState}
                            valLoss, valState = sess.run([cost, finalState], feed_dict = feed)
                            valAllLoss.append(valLoss)
                        valMeanLoss = np.mean(valAllLoss)
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "val loss: {:.3f}".format(valMeanLoss))
                        valLossHistory.append(valMeanLoss)
                    iteration += 1
                self.preProcessor.fileHandler.saveMetaFileHandler('./History/train_loss_epoch{}.txt'.format(e),
                                                                  lossHistory)
                self.preProcessor.fileHandler.saveMetaFileHandler('./History/val_loss_epoch{}.txt'.format(e),
                                                                  valLossHistory)
                saver.save(sess, "./checkpoints/sentiment.ckpt".format(e))

    def train2Category(self, epochs):
        self.intData, self.allLabel = self.preProcessor.loadPreprocessedData(self.seqL)
        cnt = 0
        for label in self.allLabel:
            if label > 0:
                cnt = cnt + 1
        print('Label higher than threshold: {} lower: {}'.format(cnt, len(self.allLabel) - cnt))
        self.splitTrainData()
        graph, inputs_, labels_, \
        keepProb, cell, initialState, \
        finalState, cost, optimizer, \
        predictions, accuracy = self.buildModel2Category(self.batchSize)

        with graph.as_default():
            saver = tf.train.Saver()

        with tf.Session(graph = graph) as sess:
            sess.run(tf.global_variables_initializer())
            iteration = 1
            lossHistory = []
            accHistory = []
            for e in range(epochs):
                state = sess.run(initialState)
                for ii, (x, y) in enumerate(self.getBatches(self.trainX, self.trainY, self.batchSize), 1):
                    feed = {inputs_: x,
                            labels_: y[:, None],
                            keepProb: 0.5,
                            initialState: state}
                    loss, state, _ = sess.run([cost, finalState, optimizer], feed_dict = feed)
                    lossHistory.append(loss)
                    if iteration % 5 == 0:
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Train loss: {:.3f}".format(loss))

                    if iteration % 25 == 0:
                        valState = sess.run(cell.zero_state(self.batchSize, tf.float32))
                        valAllAcc = []
                        for x, y in self.getBatches(self.valX, self.valY, self.batchSize):
                            feed = {inputs_: x,
                                    labels_: y[:, None],
                                    keepProb: 1,
                                    initialState: valState}
                            valAcc, valState = sess.run([accuracy, finalState], feed_dict = feed)
                            valAllAcc.append(valAcc)
                        valMeanAcc = np.mean(valAllAcc)
                        print("Epoch: {}/{}".format(e, epochs),
                              "Iteration: {}".format(iteration),
                              "Acc: {:.3f}".format(valMeanAcc))
                        accHistory.append(valMeanAcc)
                    iteration += 1
                self.preProcessor.fileHandler.saveMetaFileHandler('./History/train_loss_epoch{}.txt'.format(e),
                                                                  lossHistory)
                self.preProcessor.fileHandler.saveMetaFileHandler('./History/val_acc_epoch{}.txt'.format(e),
                                                                  accHistory)
                saver.save(sess, "./checkpoints/sentiment.ckpt".format(e))

    def predictAnime(self, synopses):
        self.preProcessor.loadVocabToInt()
        synopses = self.preProcessor.cleanUpData(synopses)
        intSynopses = self.preProcessor.converDataToInt(synopses)
        x = self.preProcessor.converIntDataToFeatures(intSynopses, self.seqL)

        graph, inputs_, labels_, \
        keepProb, cell, initialState, \
        finalState, cost, optimizer, \
        predictions, accuracy = self.buildModel2Category(len(x))

        with graph.as_default():
            saver = tf.train.Saver()
        with tf.Session(graph = graph) as sess:
            saver.restore(sess, tf.train.latest_checkpoint('checkpoints'))
            testState = sess.run(cell.zero_state(len(x), tf.float32))
            feed = {inputs_: x,
                    keepProb: 1,
                    initialState: testState}
            result, _ = sess.run([tf.round(predictions), finalState], feed_dict = feed)
            print('result:{}'.format(result))

