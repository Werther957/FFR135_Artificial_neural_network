[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadCIFAR(3);

%%Neural network 1

layers = [ ...
    imageInputLayer([32 32 3])

    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


options = trainingOptions( 'sgdm',...
'MiniBatchSize', 8192,...
'ValidationData', {xValid, tValid},...
'ValidationFrequency', 30,...
'MaxEpochs',400,...
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

%%Neural network 2

layers = [ ...
    imageInputLayer([32 32 3])

    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


options = trainingOptions( 'sgdm',...
'MiniBatchSize', 8192,...
'ValidationData', {xValid, tValid},...
'ValidationFrequency', 30,...
'MaxEpochs',400,...
'Plots', 'Training-Progress',...
'L2Regularization', 0, ...
'Momentum', 0.9, ...
'ValidationPatience', 3, ...
'Shuffle', 'every-epoch', ...
'InitialLearnRate', 0.003);

net1 = trainNetwork(xTrain, tTrain, layers, options);
train_pred1  = net1.classify(xTrain);    
valid_pred1  = net1.classify(xValid);
test_pred1  =  net1.classify(xTest);

ceTrain1 = 1 - sum(train_pred1 == tTrain)/size(tTrain,1)
ceValid1 = 1 - sum(valid_pred1 == tValid)/size(tValid,1)
ceTest1 = 1 - sum(test_pred1 == tTest)/size(tTest,1)

%%


%%Neural network 3

layers = [ ...
    imageInputLayer([32 32 3])

    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(50)
    reluLayer
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];


options = trainingOptions( 'sgdm',...
'MiniBatchSize', 8192,...
'ValidationData', {xValid, tValid},...
'ValidationFrequency', 30,...
'MaxEpochs',400,...
'Plots', 'Training-Progress',...
'L2Regularization', 0.2, ...
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
