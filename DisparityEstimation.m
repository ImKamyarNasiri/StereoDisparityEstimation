leftImage = imread('first_image.png');
rightImage = imread('second_image.png');

%convert to grayscale
leftGray = im2gray(leftImage);
%compute gradient of leftimage
leftGray_d = im2double(leftGray);
leftGray_d = imgaussfilt(leftGray_d, 3);
[gx, gy] = gradient(leftGray_d);
rightGray = im2gray(rightImage);

%hyperparameters
blockSize = 10;
y_range = 20;
searchRange = 300;
lambda = 1;

disparityMap = zeros(size(leftGray));

d_y = 0;
%iterate over rows
for i = 1: blockSize: size(leftGray, 1) - blockSize
    disp(i)
    prev_x = 0;
    prev_y = 0;
    %iterate over columns
    for j = 1: blockSize: size(leftGray, 2) - blockSize
        %search range 
        searchStart = -searchRange;
        searchEnd = searchRange;
        
        %extract left block
        blockLeft = leftGray(i:i+blockSize, j:j+blockSize);
        leftMean = mean(blockLeft(:));

        minSAD = inf;
        maxCorr = 0;
        minDisparity = 0;
        
        %search over selected range of columns
        for d = searchStart:searchEnd
            %search over selected range of rows
            % for d_y = -y_range:4:y_range
                %extract right block
                blockRight = rightGray(max(1, i+d_y):min(i+d_y+blockSize, size(rightGray, 1)), max(1, j+d):min(j+d+blockSize, size(rightGray, 2)));
                rightMean = mean(blockRight(:));
                
                if size(blockRight,1) ~= size(blockLeft,1) || size(blockRight,2) ~= size(blockLeft,2)
                    continue
                end
                
                % %monotonicity
                % for q=1:size(blockRight, 1)
                %     if ~isTrendSimilar(diff(blockLeft(q,:)), diff(blockRight(q,:)))
                %         continue
                %     end
                % end
                % 
                %normalized cross-correlation
                numerator = sum(sum(blockLeft .* blockRight));
                denominatorLeft = sum(sum(blockLeft.^2));
                denominatorRight = sum(sum(blockRight.^2));
                ncc = numerator / (sqrt(denominatorLeft * denominatorRight) + 0.0001);
                
                % sum squared error
                ssd_ = sum(sum((blockRight - blockLeft).^2));
                ssd = sum(sum((blockLeft - blockRight).^2));
                
                %smoothness terms
                if j > 1
                    prev_x = disparityMap(i, j-1);
                    smooth_loss_x = abs(d - prev_x)  * exp(-(1/blockSize)*(sum(sum(gy(i:i+blockSize, j:j+blockSize)))));
                else
                    smooth_loss_x = 0;
                end
                if i > 1
                    prev_y = disparityMap(i-1, j);
                    smooth_loss_y = abs(d - prev_y) * exp(-(1/blockSize)*(sum(sum(gx(i:i+blockSize, j:j+blockSize)))));
                else
                    smooth_loss_y = 0;
                end
                %final cost to be minimized
                Cost = 0.1 * (ssd + ssd_) + lambda * (smooth_loss_x + smooth_loss_y);
                % disp(Cost)
                uniquenessThreshold = 1000;
                %minimize cost, maximize ncc, cut the cost under threshold
                if Cost < minSAD  && ncc > maxCorr && Cost < uniquenessThreshold
                    minSAD = Cost;
                    maxCorr = ncc;
                    minDisparity = d;
                    disp_right = d_y;
                end
            % end
        end

        disparityMap(i:i+blockSize, j:j+blockSize) = abs(minDisparity);
    end
end
figure;
imshow(disparityMap, []);
colormap jet;
colorbar;
title('Depth Map');

% disparityMap = double(disparityMap);
% epsilon = 1e-2;
% disparityMap(:) = disparityMap(:) + epsilon;
% depth_map = (536.62 * 3060) ./ (disparityMap .* 100);
% disp(depth_map)
% figure;
% imshow(depth_map, []);
% colormap jet;
% colorbar;

% check monotonicity
function similar = isTrendSimilar(trend1, trend2)
    threshold = 1; 
    similar = sum(sign(trend1) == sign(trend2)) / numel(trend1) >= threshold;
end
