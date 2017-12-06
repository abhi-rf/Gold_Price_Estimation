import pandas, matplotlib.pyplot as plt, numpy
from keras.models import Sequential
from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D,Dropout,Flatten
from sklearn.preprocessing import MinMaxScaler

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


dataframe = pandas.read_csv('data.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')

#scale
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# splitting into training and testing subsets
train_size = int(len(dataset) * 0.90)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
print(len(train), len(test))

look_back = 5
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# create model
model = Sequential()
model.add(Conv1D(32, (1), input_shape=(1,5), activation='relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Conv1D(32, (1), activation='relu'))
model.add(MaxPooling1D(pool_size=(1)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(10, activation='relu'))

# Compile model
model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)


# Estimate model performance
trainScore = model.evaluate(trainX, trainY, verbose=0)
trainScore_size = len(trainScore)
trainScore = numpy.array(trainScore)
trainScore = trainScore.reshape(trainScore_size,-1)
print('Train Score: ', scaler.inverse_transform(numpy.array(trainScore)))
#print('Train Score: ', (trainScore))
testScore = model.evaluate(testX, testY, verbose=0)
testScore_size = len(testScore)
testScore = numpy.array(testScore)
testScore = testScore.reshape(testScore_size,-1)
#print('Test Score: ', scaler.inverse_transform(numpy.array([[testScore]])))
print('Test Score: ', testScore)


trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

print(scaler.inverse_transform(testPredict))
