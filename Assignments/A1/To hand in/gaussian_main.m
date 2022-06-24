% CMPE 425 - Assignment 1
% Nicholas Alderman - 20060982 - 16naa5
% October 12, 2021

load a1digits.mat

% 1.2 Training gaussian
% p(C_k) = a_k
% p(x|C_k) = (2pi*sigma^2)^(-D/2) * exp{-(1/2*sigma^2 * sum(x_i-mu_ki)^2}
% use these two values to calculate p(C_k|x)

% mu_ki = sum of each vector of D features divided by the number of
% training data points
% mu = average value for each feature in a class
mle = [];

for j = 1:10
    for i = 1:64
        mle(i,j) = sum(digits_train(i,:,j))/700
    end
end
% now have mu array for all 10 digits, need to display them with subplot &

% fix shape of mle
mle2 = reshape(mle,64,1,10)
mle2 = repmat(mle2,1,700,1)

% now need to get sigma^2
% sum of difference between feature and mle across all classes across all 
% data points across all features
% x = digits_X(i,j,k) -- j = data point #, i = feature number, k = class

s2 = sum((digits_train - mle2).^2, 'all') / (64 * 7000);

for i = 1:10
   subplot(2,5,i)
   imagesc(reshape(mle(:,i),8,8)'); axis equal; axis off; colormap gray;
end

% obtain and plot standard deviation: sigma = sqrt(sigma^2)
subplot(2,5,1)
sd = sqrt(s2);
text(10, 10, "The pixel noise standard deviation is:")
text(10,12, sprintf('%.6f',sd))

% 3 Test Performance
% gaussian test - p(C_k|x) = p(x|C_k)*p(C_k)
t1 = (2*pi*s2)^-32
t2 = (-1/(2*s2)) 

for i = 1:10
    gaussian_test(i,:,:) = ((t1 .* exp(t2 .* sum((digits_test(:,:,:) - mle(:,i)).^2))).*(1/10));
    % bayes theorem
end

% normalize so that each data point sums to 1
gaussian_test(:,:,:) = gaussian_test(:,:,:)./sum(gaussian_test)

% select most likely class for each data point
for i = 1:10
    [mx, idx] = max(gaussian_test(:,:,i), [], 1)
    gaussian_errs(i) = nnz(idx - i)
end

gaussian_errors_total = (sum(gaussian_errs) / 4000) * 100;
disp("The total error rate for gaussian classifier is:")
fprintf('Percent Error: %0.3f %%', gaussian_errors_total);