function snapped = snap(partition1, partition2)
partition1 = double(partition1);
partition2 = double(partition2);

nr1 = max(partition1(:));
nr2 = max(partition2(:));

n = hist((double(partition1(:))-1)*(nr2)+double(partition2(:)),1:nr1*nr2);
area_hist = reshape(n,nr2,nr1);


[~, max_ids] = max(area_hist,[],1);
snapped = uint8(zeros(size(partition2)));
for ii=1:nr1
    snapped(partition1==ii) = max_ids(ii)-1;
end

% int0 = area_hist(1,:);
% int1 = area_hist(2,:);
% area = int0+int1;
% percent_in = int1./area;
% area_part2 = sum(partition2(:)-1);
% 
% snapped = uint8(zeros(size(partition2)));
% for ii=1:nr1
%     % Superpixel realtive weight
%     if percent_in(ii) > 0.7
%         snapped(partition1==ii) = 1;
%     elseif percent_in(ii) > 0.3
%         area_per = int1(ii)/area_part2;
%         if area_per > 0.05
%             snapped(partition1==ii) = partition2(partition1==ii)-1;
%         end        
%     end      
% end