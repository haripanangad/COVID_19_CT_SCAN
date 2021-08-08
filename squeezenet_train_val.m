%% Initialization 
location = 'C:\Users\gox\Documents\MATLAB\augmented covidscan1'
images = imageDatastore(location,'IncludeSubfolders',1,...
    'LabelSource','foldernames');

images.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

[trainImages,valImages] = splitEachLabel(images,0.8,'randomized');

%% calling and configuring shufflenet
net = squeezenet;
lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)
lgraph = removeLayers(lgraph, {'conv10','relu_conv10','pool10','prob','ClassificationLayer_predictions'});

numClasses = numel(categories(trainImages.Labels));
newLayers = [
    convolution2dLayer(1,numClasses,'Name','newconv')
    reluLayer('Name','relu_conv10mod')
    averagePooling2dLayer(14,'Name','pool10')
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
 lgraph = addLayers(lgraph,newLayers);
 lgraph = connectLayers(lgraph,'drop9','newconv');
 %% Training options
options = trainingOptions('sgdm',...
    'MiniBatchSize',32,...
    'MaxEpochs',5,...
    'InitialLearnRate',1e-5,...
    'VerboseFrequency',1,...
    'ValidationData',valImages,...
    'ValidationPatience',Inf,...
    'Plots','training-progress',... 
    'ValidationFrequency',100);
%% Generate results
net = trainNetwork(trainImages,lgraph,options);
[YPred,probs] = classify(net,valImages);
accuracy = mean(YPred == valImages.Labels)
plotconfusion(valImages.Labels, YPred);
