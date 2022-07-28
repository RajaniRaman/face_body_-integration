
%%
im_size = 450;
im_inpath = '../data/n227';
im_dir = dir(fullfile(im_inpath, '*.png'));
im_outpath = sprintf('../data/%d_sq', im_size);
if ~exist(im_outpath, 'dir')
    mkdir(im_outpath);
end

back_image = imread('../data/BrownBgr_C.png');
back_image = double(back_image);
back_image = back_image(1:im_size, 1:im_size);

imwrite(uint8(back_image), fullfile(im_outpath, 'background_R_00.png')); 
% im_size = size(back_image, 1);

%%
for iIm = progress(1:length(im_dir))
    iIm_path = fullfile(im_dir(iIm).folder, im_dir(iIm).name);
    [im, ~, alpha] = imread(iIm_path);
    im = double(im);
    alpha = im2double(alpha);
    im_fn = pad_image(im, im_size);
    alpha_fn = pad_image(alpha, im_size);
     
%     final_image = uint8((mat2gray(double(back_image).*(1-alpha_fn) + double(im_fn).*alpha_fn))*255);
    final_image = uint8(back_image.*(1-alpha_fn) + im_fn.*alpha_fn);

    out_impath = fullfile(im_outpath,  im_dir(iIm).name);
    imwrite(final_image, out_impath)
end

%%
% helper function to pad images 
function im_fn = pad_image(im, sz)
    [wd, hg] = size(im);
    pad_size = round(([sz, sz] - [wd, hg])/2);
    pd_im = padarray(im, pad_size, 'both');
    im_fn = imresize(pd_im, [sz, sz]);
end
