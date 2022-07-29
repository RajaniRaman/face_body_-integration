%%  response of CNNs to the natural and unnatural face-body stimuli

clear all; close all;

%%%---deepnet model
network = 'alexnet_caffe';
% network = 'alexnet';
% network = 'alexnet_untrained';
% network = 'vgg16';

in_imsize = 224;
layers = ["relu1", "relu2", "relu3", "relu4", "relu5", "relu6", "relu7"];
cond = 'trained';

%- importing deepNet
switch network
    case 'alexnet_caffe'       
        caffePath ='../networks';
        protofile = fullfile(caffePath, 'alexnet_caffe_imagenet', 'alexnet_caffe_deploy.prototxt');
        datafile = fullfile(caffePath,'alexnet_caffe_imagenet','alexnet_caffe.caffemodel');  
        netw = importCaffeNetwork(protofile, datafile);
        
    case 'alexnet'
        netw = alexnet ; % matlab's alexnet
        in_imsize = 224;
        
    case 'alexnet_untrained'
        netw = load('../networks/alexnet224_untrained_1.mat');
        netw = netw.net;
        cond = untrained;
        
    case 'vgg16'  
        netw = vgg16;
        layers = ["relu1_2", "relu2_2", "relu3_3", "relu4_3", "relu5_3", "relu6", "relu7"];
end
  
% loading images and creating a dataStore object form them 
impath = '../data/500_sq';
face_imds = imageDatastore(impath, 'ReadFcn', @(x) rescaleImage(x, in_imsize));
figure; montage(face_imds, 'Indices', 115:121, 'DisplayRange', [0,5]);

% response from the deepnet
resp = cell(length(layers), 1);
for ilayer = progress(1:length(layers))
resp{ilayer} = activations(netw, face_imds, layers(ilayer),...
    'OutputAs','row', 'ExecutionEnvironment', 'cpu');
end

% save response for later use
save(sprintf('../results/data/net_resp/%s_%s_resp_from_matlab_500.mat', network, cond), 'resp');

%%%--- Helper functions
function f_img = rescaleImage(file, size)
    img = imread(file);
    img = im2double(img);
    img = imresize(img, [size, size]);
    img = repmat(img, 1, 1, 3);

   % imagenet mean and std
    Imean = [0.485, 0.456, 0.406];
    Istd = [0.229, 0.224, 0.225];  

    img_m = bsxfun(@minus, img, reshape(Imean, 1, 1, []));
    f_img = bsxfun(@rdivide, img_m, reshape(Istd, 1, 1, []));
end


function f_img = rescaleImage_nopre(file, size)
    img = imread(file);
    img = imresize(img, [size, size]);
    f_img = repmat(img, 1, 1, 3);
end


