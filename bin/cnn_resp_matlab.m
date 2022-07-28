
%%%%%%%%%%%% deepnet model%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

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

%%
% response from the deepnet
resp = cell(length(layers), 1);
for ilayer = progress(1:length(layers))
resp{ilayer} = activations(netw, face_imds, layers(ilayer),...
    'OutputAs','row', 'ExecutionEnvironment', 'cpu');
end

% save response for later use
save(sprintf('../results/data/net_resp/%s_%s_resp_from_matlab_500.mat', network, cond), 'resp');
%%

% [~, fname, ~]= cellfun(@fileparts, face_imds.Files, 'UniformOutput', false);
% mean_resp = cellfun(@(x) mean(x, 2), resp, 'UniformOutput', false);
% 
% %%
% splited_names_ = cellfun(@(x) string(split(x, '_')'), fname, 'UniformOutput', false);
% splited_names = vertcat(splited_names_{:});
% 
% %% 
% % creating table 
% layer_mat = repmat(layers, 120, 1);
% stims_mat = repmat(splited_names(:, 1), 1, length(layers));
% stim_type_mat = repmat(splited_names(:, 2), 1, length(layers));
% stim_indices_mat = repmat(splited_names(:, 3), 1, length(layers));
% resp_mat = [mean_resp{:}];
% 
% %%
% layer = layer_mat(:);
% stim = stims_mat(:);
% resp = resp_mat(:);
% stim_type = stim_type_mat(:);
% data_table2= table(layer, stim, resp, stim_type);
% 
% %%
% 
% % figure('Position', [0, 0, 1200, 200]);
% % g1= gramm('x', cellstr(data_table2.stim), 'y', data_table2.resp, 'color', cellstr(data_table2.stim_type));
% % g1.facet_grid([], cellstr(data_table2.layer), 'scale', 'independent')
% % g1.stat_summary('type', 'std', 'geom', 'bar');
% % g1.stat_summary('type', 'std', 'geom', 'black_errorbar', 'dodge', 0.5);
% % g1.draw()
% 
% %% figure 2
% face_idx = find(splited_names(:, 1)== 'MFace');
% body_idx = find(splited_names(:, 1)== 'MBody');
% Mon_idx = find(splited_names(:, 1) == 'Mon');
% 
% resp_body_plus_face = resp_mat(face_idx, :) + resp_mat(body_idx, :);
% resp_monkey = resp_mat(Mon_idx, :);
% stim_type_plus = stim_type_mat(body_idx, :);
% layer_mat2 = repmat(layers, 40, 1);
% 
% layer2 = layer_mat2(:);
% resp_m = resp_monkey(:);
% resp_bf = resp_body_plus_face(:);
% stim_type2 = stim_type_plus(:);
% 
% data_table = table(layer2, stim_type2, resp_m, resp_bf);
% data_table.('diff') = data_table.resp_m - data_table.resp_bf;
% 
% %%
% writetable(data_table, strcat('../results/data/data_table.csv'))
% 
% %%
% figure('Position', [0, 0, 1200, 400]);
% g(1, 1) = gramm('x', data_table.resp_bf, 'y', data_table.resp_m, 'color', cellstr(data_table.stim_type2));
% % g.facet_grid([],cellstr(data_table.layer2), 'scale', 'independent');
% g(1, 1).facet_grid([],cellstr(data_table.layer2));
% g(1, 1).geom_point();
% g(1, 1).geom_abline();
% g(1, 1).set_names('x', 'Response (Face+Body)', 'y', 'Response (Monkey)', 'column', '');
% % g.axe_property('XLim', [0, 1.1], 'Ylim', [0, 1.1])
% % g.draw();
% 
% 
% g(2, 1) = gramm('x', cellstr(data_table.stim_type2), 'y', data_table.diff, 'color', cellstr(data_table.stim_type2));
% % g1.facet_grid([],cellstr(data_table.layer2), 'scale', 'independent');
% g(2, 1).facet_grid([],cellstr(data_table.layer2));
% 
% g(2, 1).stat_summary('type', 'std', 'geom', 'bar');
% g(2, 1).stat_summary('type', 'std', 'geom', 'black_errorbar', 'dodge', 0.5);
% g(2, 1).set_names('y', 'M-BF', 'column', '', 'x', '');
% g.draw();
% 
% %%
% [nat_coeff, Unat_coeff] = cellfun(@(x) extract_cocoeff(x, splited_names), resp, 'UniformOutput', false);
% 
% %%
% all_nat_coeff = vertcat(nat_coeff{:});
% all_Unat_coeff = vertcat(Unat_coeff{:});
% layer3_ = repmat(layers, 20, 1);
% layer3 = cellstr(layer3_(:));
% 
% %%
% % figure
% figure('Position', [0, 0, 1200, 200]);
% g3 = gramm('x', all_nat_coeff, 'y', all_Unat_coeff);
% g3.facet_grid([], layer3)
% g3.geom_point()
% % g3.axe_property('XLim', [0, 0.4], 'YLim', [0, 0.4])
% g3.set_names('x', 'r (natural)', 'y', 'r (unnatural)')
% g3.geom_abline('intercept' , 0)
% g3.draw()
% 
% %%
% 
% function[nat_resp_monkey_z, nat_resp_sum_z, Unat_resp_monkey_z, Unat_resp_sum_z] = doZscore(nat_resp_monkey, nat_resp_sum, Unat_resp_monkey, Unat_resp_sum);
% 
%     for i = 1:20
%        mats =  zscore([nat_resp_monkey(i, :) ; nat_resp_sum(i, :); Unat_resp_monkey(i, :); Unat_resp_sum(i, :)]);
%        nat_resp_monkey_z(i, :) = mats(1, :);
%        nat_resp_sum_z(i, :) = mats(2, :);
%        Unat_resp_monkey_z(i, :) = mats(3, :);
%        Unat_resp_sum_z(i, :) = mats(4, :);   
%     end
% 
% end
% 
% function [nat_coeff, Unat_coeff] = extract_cocoeff(resp, splited_names)
%     resp( :, all(~resp,1) ) = [];
%     st_type = splited_names(:, 2);
%     st = splited_names(:, 1);
%     nat_resp_monkey = resp(strcmp(st_type, 'R') & strcmp(st, 'Mon'), :);  % response to natural
%     nat_resp_face = resp(strcmp(st_type, 'R') & strcmp(st, 'MFace'), :);  % response to natural
%     nat_resp_body = resp(strcmp(st_type, 'R') & strcmp(st, 'MBody'), :);  % response to natural
%     nat_resp_sum = nat_resp_body+nat_resp_face;
% 
%     Unat_resp_monkey = resp(strcmp(st_type, 'Irr') & strcmp(st, 'Mon'), :); % response to unatural 
%     Unat_resp_face = resp(strcmp(st_type, 'Irr') & strcmp(st, 'MFace'), :); % response to unatural 
%     Unat_resp_body = resp(strcmp(st_type, 'Irr') & strcmp(st, 'MBody'), :); % response to unatural 
%     Unat_resp_sum = Unat_resp_body+Unat_resp_face;
% 
%     [nat_resp_monkey_z, nat_resp_sum_z, Unat_resp_monkey_z, Unat_resp_sum_z] =...\
%         doZscore(nat_resp_monkey, nat_resp_sum, Unat_resp_monkey, Unat_resp_sum);
% 
%     nat_coeff = diag(corr(nat_resp_monkey_z', nat_resp_sum_z'));
%     Unat_coeff = diag(corr(Unat_resp_monkey_z', Unat_resp_sum_z'));
% end
% 
% %%


% function features = getFeatures(imds)
%     str = string(imds.Files);
%     features = string(zeros(numel(str), 9));
%     for i= 1:numel(str)
%         [~, fName, ~] = fileparts(str(i));
%        features(i, :)=  [strsplit(fName, '_')];
%     end   
% end

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


