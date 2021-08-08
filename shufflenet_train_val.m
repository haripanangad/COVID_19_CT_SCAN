%% Initialization 
location = 'C:\Users\gox\Documents\MATLAB\augmented covidscan1'
images = imageDatastore(location,'IncludeSubfolders',1,...
    'LabelSource','foldernames');
images.ReadFcn = @(loc)imresize(imread(loc),[224,224]);

[trainImages,valImages] = splitEachLabel(images,0.8,'randomized');
%% calling and configuring shufflenet
net = shufflenet;
lgraph = layerGraph(net);
figure('Units','normalized','Position',[0.1 0.1 0.8 0.8]);
plot(lgraph)
%lgraph = removeLayers(lgraph, {'conv19','avg1','softmax','output'});
lgraph = removeLayers(lgraph, {'node_200','node_202','node_203','ClassificationLayer_node_203'});
numClasses = numel(categories(trainImages.Labels));
newLayers = [
    fullyConnectedLayer(numClasses,'Name','fc','WeightLearnRateFactor',3,'BiasLearnRateFactor', 2);
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'node_199','fc');
%% Training options
options = trainingOptions('sgdm',...
    'MiniBatchSize',32,...
    'MaxEpochs',5,...
    'InitialLearnRate',1e-5,...
    'Plots','training-progress',...
    'VerboseFrequency',1,...
    'ValidationData',valImages,...
    'L2Regularization', 0.1,...
    'ValidationPatience',Inf,...
    'Plots','training-progress',... 
    'ValidationFrequency',100);

%% Generate results
net = trainNetwork(trainImages,lgraph,options);
[YPred,probs] = classify(net,valImages);
accuracy = mean(YPred == valImages.Labels)
confMat = confusionmat(valImages.Labels, YPred);
plotconfusion(valImages.Labels, YPred);
