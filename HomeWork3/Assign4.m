[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(4);

%%Neural network 1

layers = [ ...
    imageInputLayer([32 32 3])
    
    convolution2dLayer([5 5], 20,'Stride',[1 1], 'Padding', [1 1 1 1])
    reluLayer
    
    maxPooling2dLayer([2 2], 'Stride', [2 2],'Padding', [0 0 0 0])
    
    fullyConnectedLayer(50)
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];



options = trainingOptions( 'sgdm',...
'MiniBatchSize', 8192,...
'ValidationData', {xValid, tValid},...
'ValidationFrequency', 30,...
'MaxEpochs',120,...
'Plots', 'Training-Progress',...
'L2Regularization', 0, ...
'Momentum', 0.9, ...
'ValidationPatience', 3, ...
'Shuffle', 'every-epoch', ...
'InitialLearnRate', 0.001);
%%%%%%'ExecutionEnvironment', 'parallel' notwork

net1 = trainNetwork(xTrain, tTrain, layers, options);
train_pred1  = net1.classify(xTrain);    
valid_pred1  = net1.classify(xValid);
test_pred1  =  net1.classify(xTest);

ceTrain1 = 1 - sum(train_pred1 == tTrain)/size(tTrain,1)
ceValid1 = 1 - sum(valid_pred1 == tValid)/size(tValid,1)
ceTest1 = 1 - sum(test_pred1 == tTest)/size(tTest,1)
%%

%%Neural network 2
layers = [ ...
    imageInputLayer([32 32 3])
    
    convolution2dLayer([3 3], 20,'Stride',[1 1], 'Padding', [1 1 1 1])
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', [2 2],'Padding', [0 0 0 0])
    
    convolution2dLayer([3 3], 30,'Stride',[1 1], 'Padding', [1 1 1 1])
    batchNormalizationLayer
    reluLayer
    maxPooling2dLayer([2 2], 'Stride', [2 2],'Padding', [0 0 0 0])
    
    convolution2dLayer([3 3], 50,'Stride',[1 1], 'Padding', [1 1 1 1])
    batchNormalizationLayer
    reluLayer

    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];



options = trainingOptions( 'sgdm',...
'MiniBatchSize', 8192,...
'ValidationData', {xValid, tValid},...
'ValidationFrequency', 30,...
'MaxEpochs',120,...
'Plots', 'Training-Progress',...
'L2Regularization', 0, ...
'Momentum', 0.9, ...
'ValidationPatience', 3, ...
'Shuffle', 'every-epoch', ...
'InitialLearnRate', 0.001);

net1 = trainNetwork(xTrain, tTrain, layers, options);
train_pred1  = net1.classify(xTrain);    
valid_pred1  = net1.classify(xValid);
test_pred1  =  net1.classify(xTest);

ceTrain1 = 1 - sum(train_pred1 == tTrain)/size(tTrain,1)
ceValid1 = 1 - sum(valid_pred1 == tValid)/size(tValid,1)
ceTest1 = 1 - sum(test_pred1 == tTest)/size(tTest,1)

%%

