
%% Initialization 
location = 'C:\Users\gox\Documents\MATLAB\augmented covidscan1'
images = imageDatastore(location,'IncludeSubfolders',1,...
    'LabelSource','foldernames');
images.ReadFcn = @(loc)imresize(imread(loc),[227,227]);

 [trainImages,valImages] = splitEachLabel(images,0.8,'randomized');

 inputSize = [227,227,3];
 channels = 3; 

% net1 = squeezenet

%% Calling pre-trained model layers with weights
 net = shufflenet;
 

  lgraph = layerGraph(net);
  %params = load("C:\Users\gox\Documents\MATLAB\New folder (2)\params_2020_11_22__10_58_02.mat");
  params = load("C:\Users\ACER\Documents\MATLAB\New Folder (2)\params_2020_11_22__10_58_02.mat");
  
  tempLayers = [
    imageInputLayer([227 227 3],"Name","data","Mean",params.data.Mean)
    convolution2dLayer([3 3],64,"Name","conv1","Stride",[2 2],"Bias",params.conv1.Bias,"Weights",params.conv1.Weights)
    reluLayer("Name","relu_conv1")
    maxPooling2dLayer([3 3],"Name","pool1","Stride",[2 2])
    convolution2dLayer([1 1],16,"Name","fire2-squeeze1x1","Bias",params.fire2_squeeze1x1.Bias,"Weights",params.fire2_squeeze1x1.Weights)
    reluLayer("Name","fire2-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire2-expand1x1","Bias",params.fire2_expand1x1.Bias,"Weights",params.fire2_expand1x1.Weights)
    reluLayer("Name","fire2-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire2-expand3x3","Padding",[1 1 1 1],"Bias",params.fire2_expand3x3.Bias,"Weights",params.fire2_expand3x3.Weights)
    reluLayer("Name","fire2-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire2-concat")
    convolution2dLayer([1 1],16,"Name","fire3-squeeze1x1","Bias",params.fire3_squeeze1x1.Bias,"Weights",params.fire3_squeeze1x1.Weights)
    reluLayer("Name","fire3-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],64,"Name","fire3-expand3x3","Padding",[1 1 1 1],"Bias",params.fire3_expand3x3.Bias,"Weights",params.fire3_expand3x3.Weights)
    reluLayer("Name","fire3-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],64,"Name","fire3-expand1x1","Bias",params.fire3_expand1x1.Bias,"Weights",params.fire3_expand1x1.Weights)
    reluLayer("Name","fire3-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire3-concat")
    maxPooling2dLayer([3 3],"Name","pool3","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],32,"Name","fire4-squeeze1x1","Bias",params.fire4_squeeze1x1.Bias,"Weights",params.fire4_squeeze1x1.Weights)
    reluLayer("Name","fire4-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire4-expand1x1","Bias",params.fire4_expand1x1.Bias,"Weights",params.fire4_expand1x1.Weights)
    reluLayer("Name","fire4-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire4-expand3x3","Padding",[1 1 1 1],"Bias",params.fire4_expand3x3.Bias,"Weights",params.fire4_expand3x3.Weights)
    reluLayer("Name","fire4-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire4-concat")
    convolution2dLayer([1 1],32,"Name","fire5-squeeze1x1","Bias",params.fire5_squeeze1x1.Bias,"Weights",params.fire5_squeeze1x1.Weights)
    reluLayer("Name","fire5-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],128,"Name","fire5-expand1x1","Bias",params.fire5_expand1x1.Bias,"Weights",params.fire5_expand1x1.Weights)
    reluLayer("Name","fire5-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],128,"Name","fire5-expand3x3","Padding",[1 1 1 1],"Bias",params.fire5_expand3x3.Bias,"Weights",params.fire5_expand3x3.Weights)
    reluLayer("Name","fire5-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire5-concat")
    maxPooling2dLayer([3 3],"Name","pool5","Padding",[0 1 0 1],"Stride",[2 2])
    convolution2dLayer([1 1],48,"Name","fire6-squeeze1x1","Bias",params.fire6_squeeze1x1.Bias,"Weights",params.fire6_squeeze1x1.Weights)
    reluLayer("Name","fire6-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire6-expand1x1","Bias",params.fire6_expand1x1.Bias,"Weights",params.fire6_expand1x1.Weights)
    reluLayer("Name","fire6-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire6-expand3x3","Padding",[1 1 1 1],"Bias",params.fire6_expand3x3.Bias,"Weights",params.fire6_expand3x3.Weights)
    reluLayer("Name","fire6-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire6-concat")
    convolution2dLayer([1 1],48,"Name","fire7-squeeze1x1","Bias",params.fire7_squeeze1x1.Bias,"Weights",params.fire7_squeeze1x1.Weights)
    reluLayer("Name","fire7-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],192,"Name","fire7-expand1x1","Bias",params.fire7_expand1x1.Bias,"Weights",params.fire7_expand1x1.Weights)
    reluLayer("Name","fire7-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],192,"Name","fire7-expand3x3","Padding",[1 1 1 1],"Bias",params.fire7_expand3x3.Bias,"Weights",params.fire7_expand3x3.Weights)
    reluLayer("Name","fire7-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire7-concat")
    convolution2dLayer([1 1],64,"Name","fire8-squeeze1x1","Bias",params.fire8_squeeze1x1.Bias,"Weights",params.fire8_squeeze1x1.Weights)
    reluLayer("Name","fire8-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire8-expand3x3","Padding",[1 1 1 1],"Bias",params.fire8_expand3x3.Bias,"Weights",params.fire8_expand3x3.Weights)
    reluLayer("Name","fire8-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire8-expand1x1","Bias",params.fire8_expand1x1.Bias,"Weights",params.fire8_expand1x1.Weights)
    reluLayer("Name","fire8-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire8-concat")
    convolution2dLayer([1 1],64,"Name","fire9-squeeze1x1","Bias",params.fire9_squeeze1x1.Bias,"Weights",params.fire9_squeeze1x1.Weights)
    reluLayer("Name","fire9-relu_squeeze1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([3 3],256,"Name","fire9-expand3x3","Padding",[1 1 1 1],"Bias",params.fire9_expand3x3.Bias,"Weights",params.fire9_expand3x3.Weights)
    reluLayer("Name","fire9-relu_expand3x3")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    convolution2dLayer([1 1],256,"Name","fire9-expand1x1","Bias",params.fire9_expand1x1.Bias,"Weights",params.fire9_expand1x1.Weights)
    reluLayer("Name","fire9-relu_expand1x1")];
lgraph = addLayers(lgraph,tempLayers);

tempLayers = [
    depthConcatenationLayer(2,"Name","fire9-concat")
    dropoutLayer(0.5,"Name","drop9")
    convolution2dLayer([1 1],1000,"Name","conv10","Bias",params.conv10.Bias,"Weights",params.conv10.Weights)
    reluLayer("Name","relu_conv10")
    globalAveragePooling2dLayer("Name","pool10")
    softmaxLayer("Name","prob")
    classificationLayer("Name","ClassificationLayer_predictions","Classes",params.ClassificationLayer_predictions.Classes)];
lgraph = addLayers(lgraph,tempLayers);

%% Connections between layers
lgraph = connectLayers(lgraph,"fire2-relu_squeeze1x1","fire2-expand1x1");
lgraph = connectLayers(lgraph,"fire2-relu_squeeze1x1","fire2-expand3x3");
lgraph = connectLayers(lgraph,"fire2-relu_expand1x1","fire2-concat/in1");
lgraph = connectLayers(lgraph,"fire2-relu_expand3x3","fire2-concat/in2");
lgraph = connectLayers(lgraph,"fire3-relu_squeeze1x1","fire3-expand3x3");
lgraph = connectLayers(lgraph,"fire3-relu_squeeze1x1","fire3-expand1x1");
lgraph = connectLayers(lgraph,"fire3-relu_expand3x3","fire3-concat/in2");
lgraph = connectLayers(lgraph,"fire3-relu_expand1x1","fire3-concat/in1");
lgraph = connectLayers(lgraph,"fire4-relu_squeeze1x1","fire4-expand1x1");
lgraph = connectLayers(lgraph,"fire4-relu_squeeze1x1","fire4-expand3x3");
lgraph = connectLayers(lgraph,"fire4-relu_expand1x1","fire4-concat/in1");
lgraph = connectLayers(lgraph,"fire4-relu_expand3x3","fire4-concat/in2");
lgraph = connectLayers(lgraph,"fire5-relu_squeeze1x1","fire5-expand1x1");
lgraph = connectLayers(lgraph,"fire5-relu_squeeze1x1","fire5-expand3x3");
lgraph = connectLayers(lgraph,"fire5-relu_expand1x1","fire5-concat/in1");
lgraph = connectLayers(lgraph,"fire5-relu_expand3x3","fire5-concat/in2");
lgraph = connectLayers(lgraph,"fire6-relu_squeeze1x1","fire6-expand1x1");
lgraph = connectLayers(lgraph,"fire6-relu_squeeze1x1","fire6-expand3x3");
lgraph = connectLayers(lgraph,"fire6-relu_expand1x1","fire6-concat/in1");
lgraph = connectLayers(lgraph,"fire6-relu_expand3x3","fire6-concat/in2");
lgraph = connectLayers(lgraph,"fire7-relu_squeeze1x1","fire7-expand1x1");
lgraph = connectLayers(lgraph,"fire7-relu_squeeze1x1","fire7-expand3x3");
lgraph = connectLayers(lgraph,"fire7-relu_expand1x1","fire7-concat/in1");
lgraph = connectLayers(lgraph,"fire7-relu_expand3x3","fire7-concat/in2");
lgraph = connectLayers(lgraph,"fire8-relu_squeeze1x1","fire8-expand3x3");
lgraph = connectLayers(lgraph,"fire8-relu_squeeze1x1","fire8-expand1x1");
lgraph = connectLayers(lgraph,"fire8-relu_expand3x3","fire8-concat/in2");
lgraph = connectLayers(lgraph,"fire8-relu_expand1x1","fire8-concat/in1");
lgraph = connectLayers(lgraph,"fire9-relu_squeeze1x1","fire9-expand3x3");
lgraph = connectLayers(lgraph,"fire9-relu_squeeze1x1","fire9-expand1x1");
lgraph = connectLayers(lgraph,"fire9-relu_expand3x3","fire9-concat/in2");
lgraph = connectLayers(lgraph,"fire9-relu_expand1x1","fire9-concat/in1");

 % lgraph =addLayers(lgraph, net1.Layers);
   lgraph = removeLayers(lgraph, {'data','conv10','relu_conv10','pool10','prob','ClassificationLayer_predictions'});
   lgraph = removeLayers(lgraph, {'Input_gpu_0|data_0','node_200','node_202','node_203','ClassificationLayer_node_203'});
  input_layer = imageInputLayer(inputSize,'Name','Input_Layer');
lgraph = addLayers(lgraph,input_layer);


%% Splitting input into two streams
for k = 1 : 6
    if (k <=3) 
      %  i = 1;
    eval(sprintf('ch_%d_splitter = convolution2dLayer(1,1,''Name'',''channel_%d_splitter'',''WeightLearnRateFactor'',0,''BiasLearnRateFactor'',0,''WeightL2Factor'',0,''BiasL2Factor'',0);',k,k));
    eval(sprintf('ch_%d_splitter.Weights = zeros(1,1,channels,1);',k));
    eval(sprintf('ch_%d_splitter.Weights(1,1,%d,1) = 1;',k,k));
    eval(sprintf('ch_%d_splitter.Bias = zeros(1,1,1,1);',k));
    eval(sprintf('lgraph = addLayers(lgraph,ch_%d_splitter);',k));
    eval(sprintf('lgraph = connectLayers(lgraph,''Input_Layer'',''channel_%d_splitter'');',k));
   
         % i = 2;
    eval(sprintf('ch1_%d_splitter = convolution2dLayer(1,1,''Name'',''channel1_%d_splitter'',''WeightLearnRateFactor'',0,''BiasLearnRateFactor'',0,''WeightL2Factor'',0,''BiasL2Factor'',0);',k,k));
    eval(sprintf('ch1_%d_splitter.Weights = zeros(1,1,channels,1);',k));
    eval(sprintf('ch1_%d_splitter.Weights(1,1,%d,1) = 1;',k,k));
    eval(sprintf('ch1_%d_splitter.Bias = zeros(1,1,1,1);',k));
    eval(sprintf('lgraph = addLayers(lgraph,ch1_%d_splitter);',k));
    eval(sprintf('lgraph = connectLayers(lgraph,''Input_Layer'',''channel1_%d_splitter'');',k));
    end
end
input_stream_1 = depthConcatenationLayer(3,'Name','input_stream_1');
lgraph = addLayers(lgraph,input_stream_1);
input_stream_2 = depthConcatenationLayer(3,'Name','input_stream_2');
lgraph = addLayers(lgraph,input_stream_2);
lgraph = connectLayers(lgraph,'channel_1_splitter','input_stream_1/in1');
lgraph = connectLayers(lgraph,'channel_2_splitter','input_stream_1/in2');
lgraph = connectLayers(lgraph,'channel_3_splitter','input_stream_1/in3');
lgraph = connectLayers(lgraph,'channel1_1_splitter','input_stream_2/in1');
lgraph = connectLayers(lgraph,'channel1_2_splitter','input_stream_2/in2');
lgraph = connectLayers(lgraph,'channel1_3_splitter','input_stream_2/in3');

%% combining features from shufflenet and squeezenet
  shufflemaxpool = maxPooling2dLayer(4,'Stride',1,'Name','shufflemaxpool');
     lgraph = addLayers(lgraph,shufflemaxpool);
     lgraph = connectLayers(lgraph,'input_stream_1','shufflemaxpool');
  squeezmaxpool = maxPooling2dLayer(2,'Stride',2,'Name','squeezmaxpool');
     lgraph = addLayers(lgraph,squeezmaxpool);
     lgraph = connectLayers(lgraph,'drop9','squeezmaxpool');   
  lgraph = connectLayers(lgraph,'shufflemaxpool','node_1');
  lgraph = connectLayers(lgraph,'input_stream_2','conv1');
  fcdepth = depthConcatenationLayer(2,'Name','fcdepth');
lgraph = addLayers(lgraph,fcdepth);
lgraph = connectLayers(lgraph,'node_199','fcdepth/in1');
lgraph = connectLayers(lgraph,'squeezmaxpool','fcdepth/in2');
newLayers = [
    fullyConnectedLayer(1024,'Name','fc1','WeightLearnRateFactor',8,'BiasLearnRateFactor', 2);
    fullyConnectedLayer(1024,'Name','fc2','WeightLearnRateFactor',8,'BiasLearnRateFactor', 2);
     fullyConnectedLayer(2,'Name','fc','WeightLearnRateFactor',8,'BiasLearnRateFactor', 2);
    softmaxLayer('Name','softmax')
    classificationLayer('Name','classoutput')];
lgraph = addLayers(lgraph,newLayers);
lgraph = connectLayers(lgraph,'fcdepth','fc1');
 plot(lgraph)
% 

%% Training and validation
 options = trainingOptions('sgdm',...
    'MiniBatchSize',32,...
    'MaxEpochs',5,...
     'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.2, ...
    'LearnRateDropPeriod',3, ...
    'InitialLearnRate',1e-5,...
    'Plots','training-progress',...
    'VerboseFrequency',1,...
    'ValidationData',valImages,...
    'L2Regularization', 0.0001,...
    'ValidationPatience',Inf,...
    'Momentum', 0.9,...
    'Plots','training-progress',... 
    'ValidationFrequency',100);

net = trainNetwork(trainImages,lgraph,options);


%% Output result
[YPred,probs] = classify(net,valImages);
accuracy = mean(YPred == valImages.Labels)
confMat = confusionmat(valImages.Labels, YPred);
plotconfusion(valImages.Labels, YPred);
